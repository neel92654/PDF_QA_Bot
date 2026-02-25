from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import re
import uuid
import uvicorn
import torch
import time
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from slowapi import Limiter
from slowapi.util import get_remote_address

# Post-processing helper: strips prompt echoes / context leakage from LLM output
# so that the API always returns only the clean, user-facing answer.
from utils.postprocess import extract_final_answer

load_dotenv()

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ---------------------------------------------------------------------------
# SESSION MANAGEMENT
# Format: { session_id: { "docs": { doc_id: FAISS }, "last_accessed": float } }
# ---------------------------------------------------------------------------
sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour

HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-base")
generation_tokenizer = None
generation_model = None
generation_is_encoder_decoder = False

# Load local embedding model once at startup
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# STARTUP: Load generation model ONCE
# ---------------------------------------------------------------------------
print("[STARTUP] Loading generation model at startup...")
config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
generation_is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
generation_tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

if generation_is_encoder_decoder:
    generation_model = AutoModelForSeq2SeqLM.from_pretrained(
        HF_GENERATION_MODEL,
        low_cpu_mem_usage=False
    )
else:
    generation_model = AutoModelForCausalLM.from_pretrained(
        HF_GENERATION_MODEL,
        low_cpu_mem_usage=False
    )

if torch.cuda.is_available():
    generation_model = generation_model.to("cuda")
    print("[GPU] Model loaded on CUDA")
else:
    print("[CPU] Model loaded on CPU")

generation_model.eval()
print("[STARTUP] ✅ Model loaded successfully!")


# ---------------------------------------------------------------------------
# TEXT NORMALIZATION UTILITIES
# ---------------------------------------------------------------------------

def normalize_spaced_text(text: str) -> str:
    """
    Fixes character-level spaced text produced by PyPDFLoader on certain
    vector-based PDFs (e.g. NPTEL / IBM Coursera certificates).
    """
    def fix_spaced_word(match):
        return match.group(0).replace(" ", "")

    pattern = r'\b(?:[A-Za-z] ){2,}[A-Za-z]\b'
    return re.sub(pattern, fix_spaced_word, text)




# ---------------------------------------------------------------------------
# MODEL GENERATION
# ---------------------------------------------------------------------------

def generate_response(prompt: str, max_new_tokens: int) -> str:
    """Run inference with the globally loaded model (no reload per request)."""
    global generation_tokenizer, generation_model, generation_is_encoder_decoder
    tokenizer = generation_tokenizer
    model = generation_model
    is_encoder_decoder = generation_is_encoder_decoder
    model_device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    if is_encoder_decoder:
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return text.strip()

    input_len = encoded["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------------------

class PDFPath(BaseModel):
    filePath: str
    session_id: str

class AskRequest(BaseModel):
    question: str
    session_id: str
    doc_ids: list = []       # optional list of doc IDs to restrict search
    history: list = []

class SummarizeRequest(BaseModel):
    pdf: str | None = None
    session_id: str
    doc_ids: list = []

class CompareRequest(BaseModel):
    session_id: str
    doc_ids: list = []       # must contain exactly 2 (or more) doc IDs


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def cleanup_expired_sessions():
    current_time = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if current_time - data["last_accessed"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del sessions[sid]


def get_session_docs(session_id: str, doc_ids: list) -> list[FAISS]:
    """
    Return a list of FAISS vectorstores from a session.
    If doc_ids is provided, restrict to those; otherwise return all docs.
    """
    session_data = sessions.get(session_id)
    if not session_data:
        return []
    all_docs = session_data.get("docs", {})
    if doc_ids:
        return [all_docs[d] for d in doc_ids if d in all_docs]
    return list(all_docs.values())


def merged_similarity_search(vectorstores: list[FAISS], query: str, k: int = 4) -> list:
    """
    Perform similarity search across multiple FAISS stores and return
    the top-k chunks merged from all stores (de-duplicated by content).
    """
    seen = set()
    results = []
    per_store_k = max(k, k * 2 // max(len(vectorstores), 1))
    for vs in vectorstores:
        for doc in vs.similarity_search(query, k=per_store_k):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                results.append(doc)
    return results[:k * len(vectorstores)]  # return more chunks for multi-doc


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/process-pdf")
@limiter.limit("15/15 minutes")
def process_pdf(request: Request, data: PDFPath):
    cleanup_expired_sessions()

    loader = PyPDFLoader(data.filePath)
    raw_docs = loader.load()

    # Normalize at ingestion
    cleaned_docs = [
        Document(
            page_content=normalize_spaced_text(doc.page_content),
            metadata=doc.metadata
        )
        for doc in raw_docs
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(cleaned_docs)
    if not chunks:
        return {"error": "No text chunks generated from the PDF. Please check your file."}

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Generate a unique doc_id for this PDF
    doc_id = str(uuid.uuid4())

    if data.session_id not in sessions:
        sessions[data.session_id] = {"docs": {}, "last_accessed": time.time()}

    sessions[data.session_id]["docs"][doc_id] = vectorstore
    sessions[data.session_id]["last_accessed"] = time.time()

    return {"message": "PDF processed successfully", "doc_id": doc_id}


@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    cleanup_expired_sessions()

    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"answer": "Session expired or no PDF uploaded for this session."}

    session_data["last_accessed"] = time.time()

    vectorstores = get_session_docs(data.session_id, data.doc_ids)
    if not vectorstores:
        return {"answer": "No documents found for the selected session."}

    question = data.question
    history = data.history

    # Build conversation context (last 5 turns max)
    conversation_context = ""
    for msg in history[-5:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation_context += f"{role}: {content}\n"

    # Retrieve relevant chunks from all selected documents
    docs = merged_similarity_search(vectorstores, question, k=4)
    if not docs:
        return {"answer": "No relevant context found in the selected documents."}

    context = "\n\n".join([doc.page_content for doc in docs])

    # ── Tight prompt: answer-only, no echoing ────────────────────────────────
    conv_block = (
        f"Previous conversation:\n{conversation_context.strip()}\n\n"
        if conversation_context.strip() else ""
    )

    prompt = (
        "You are a precise question-answering assistant.\n"
        "Read the document excerpt below and answer the question using ONLY the provided information.\n"
        "Your response must be a short, direct answer with no extra explanation.\n"
        "Do NOT repeat the question, context, or any instructions.\n\n"
        f"{conv_block}"
        f"Document excerpt:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    raw_answer = generate_response(prompt, max_new_tokens=150)
    # Post-process: remove prompt echoes / context leakage; return clean answer only.
    clean_answer = extract_final_answer(raw_answer)
    return {"answer": clean_answer}


@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    cleanup_expired_sessions()

    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"summary": "Session expired or no PDF uploaded for this session."}

    session_data["last_accessed"] = time.time()

    vectorstores = get_session_docs(data.session_id, data.doc_ids)
    if not vectorstores:
        return {"summary": "No documents found for the selected session."}

    docs = merged_similarity_search(vectorstores, "Give a concise summary of the document.", k=6)
    if not docs:
        return {"summary": "No document context available to summarize."}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = (
        "Summarize the following document excerpt in 5-7 clear bullet points.\n"
        "Each bullet point must state one key fact (who, what, when, where, or why).\n"
        "Use ONLY the information provided below. Do NOT add assumptions.\n"
        "Do NOT repeat these instructions in your response.\n\n"
        f"Document excerpt:\n{context}\n\n"
        "Summary:"
    )

    raw_summary = generate_response(prompt, max_new_tokens=300)
    # Post-process: strip any leaked prompt sections from the generated summary.
    summary = extract_final_answer(raw_summary)
    return {"summary": summary}


@app.post("/compare")
@limiter.limit("10/15 minutes")
def compare_documents(request: Request, data: CompareRequest):
    cleanup_expired_sessions()

    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"comparison": "Session expired or no PDFs uploaded for this session."}

    if len(data.doc_ids) < 2:
        return {"comparison": "Please select at least 2 documents to compare."}

    session_data["last_accessed"] = time.time()

    vectorstores = get_session_docs(data.session_id, data.doc_ids)
    if len(vectorstores) < 2:
        return {"comparison": "Could not find enough documents in the session. Please re-upload."}

    # Retrieve top chunks from each document separately for fair comparison
    query = "summarize the main topic, purpose, and key details of this document"
    per_doc_contexts = []
    for i, vs in enumerate(vectorstores):
        chunks = vs.similarity_search(query, k=4)
        text = "\n".join([c.page_content for c in chunks])
        per_doc_contexts.append(f"Document {i + 1}:\n{text}")

    combined_context = "\n\n---\n\n".join(per_doc_contexts)

    prompt = (
        "You are a document comparison assistant.\n"
        "Compare the documents below and produce a structured comparison with:\n"
        "1. A brief overview of each document.\n"
        "2. Key similarities.\n"
        "3. Key differences.\n"
        "Base your comparison ONLY on the provided excerpts. Do NOT invent information.\n"
        "Do NOT repeat these instructions.\n\n"
        f"{combined_context}\n\n"
        "Comparison:"
    )

    raw = generate_response(prompt, max_new_tokens=400)
    # Post-process: ensure comparison output contains no leaked prompt text.
    comparison = extract_final_answer(raw)
    return {"comparison": comparison}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
