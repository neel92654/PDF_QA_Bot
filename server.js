const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const rateLimit = require("express-rate-limit");

let chatHistory = [];
const app = express();
app.use(cors());
app.set('trust proxy', 1); // Fix ERR_ERL_UNEXPECTED_X_FORWARDED_FOR
app.use(express.json());

// Rate limiting middleware
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: "Too many PDF uploads from this IP, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

const askLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
  message: "Too many questions asked, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

const summarizeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: "Too many summarization requests, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

// Storage for uploaded PDFs
const upload = multer({ dest: "uploads/" });

// Route: Upload PDF
app.post("/upload", uploadLimiter, upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded. Use form field name 'file'." });
    }

    const filePath = path.join(__dirname, req.file.path);

    // Send PDF to Python service
    await axios.post("http://localhost:5000/process-pdf", {
      filePath: filePath,
    });

    res.json({ message: "PDF uploaded & processed successfully!" });
  } catch (err) {
    const details = err.response?.data || err.message;
    console.error("Upload processing failed:", details);
    res.status(500).json({ error: "PDF processing failed", details });
  }
});

// Route: Ask Question
app.post("/ask", askLimiter, async (req, res) => {
  try {
    const question = req.body.question;

    // Add user message to history
    chatHistory.push({
      role: "user",
      content: question
    });

    // Send question + history to FastAPI
    const response = await axios.post(
      "http://localhost:5000/ask",
      {
        question: question,
        history: chatHistory
      }
    );

    // Add assistant response to history
    chatHistory.push({
      role: "assistant",
      content: response.data.answer
    });

    res.json(response.data);

  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Error asking question" });
  }
});

app.post("/clear-history", (req, res) => {
  chatHistory = [];
  res.json({ message: "History cleared" });
});

app.post("/summarize", summarizeLimiter, async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/summarize", req.body || {});
    res.json({ summary: response.data.summary });
  } catch (err) {
    const details = err.response?.data || err.message;
    console.error("Summarization failed:", details);
    res.status(500).json({ error: "Error summarizing PDF", details });
  }
});

app.listen(4000, () => console.log("Backend running on http://localhost:4000"));
