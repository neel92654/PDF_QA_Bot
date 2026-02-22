const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const rateLimit = require("express-rate-limit");
const crypto = require("crypto");
const { fileTypeFromFile } = require("file-type");

const app = express();
app.set("trust proxy", 1); // Trust first proxy for rate limiting if behind a proxy
app.use(cors());
app.use(express.json());

// Rate limiting middleware
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Limit each IP to 5 upload requests per windowMs
  message: "Too many PDF uploads from this IP, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
});

const askLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 30, // Limit each IP to 30 questions per windowMs
  message: "Too many questions asked, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
});

const summarizeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10, // Limit each IP to 10 summarizations per windowMs
  message: "Too many summarization requests, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
});

// Storage for uploaded PDFs
const UPLOAD_DIR = path.resolve(__dirname, "uploads");

if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR);
}

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, UPLOAD_DIR);
    },
    filename: (req, file, cb) => {
      const safeName = crypto.randomUUID();
      cb(null, `${safeName}.pdf`);
    },
  }),
  limits: {
    fileSize: 20 * 1024 * 1024, // 20MB
  },
}); 

// Route: Upload PDF
app.post("/upload", uploadLimiter, upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded. Use form field name 'file'." });
    }

    const filePath = path.resolve(req.file.path);

    //Magic byte check to ensure it's a PDF
    const fileType = await fileTypeFromFile(filePath);
    if (!fileType || fileType.mime !== "application/pdf") {
      fs.unlinkSync(filePath); // Delete the invalid file
      return res.status(400).json({ error: "Invalid PDF file uploaded." });
    }

    //Ensure file is not empty
    const stats = fs.statSync(filePath);
    if (stats.size === 0) {
      fs.unlinkSync(filePath); // Delete the empty file
      return res.status(400).json({ error: "Uploaded PDF is empty." });
    }

    //Ensure file stays in uploads directory and is not executable
    if (!filePath.startsWith(UPLOAD_DIR) || path.extname(filePath) !== ".pdf") {
      fs.unlinkSync(filePath); // Delete the suspicious file
      return res.status(400).json({ error: "Invalid file path or type." });
    }

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
  const { question } = req.body;
  try {
    const response = await axios.post("http://localhost:5000/ask", {
      question,
    });

    res.json({ answer: response.data.answer });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "Error answering question" });
  }
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

// Global error handler for multer file size limit
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({ error: "File exceeds 20MB limit." });
    }
  }
  next(err);
});

app.listen(4000, () => console.log("Backend running on http://localhost:4000"));
