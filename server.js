/* =====================================================
   LEGAL DOCUMENT DECODER - EXPRESS SERVER
   Handles document upload + AI-powered Q&A via Groq
   Analysis is done client-side for instant results
   ===================================================== */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const { v4: uuidv4 } = require('uuid');
const Groq = require('groq-sdk');
const rateLimit = require('express-rate-limit');
const path = require('path');
const prompts = require('./prompts');

const app = express();
const PORT = process.env.PORT || 3000;

// =====================================================
// SECURITY: Validate API key exists at startup
// =====================================================
let groq = null;
if (!process.env.GROQ_API_KEY) {
    console.warn('WARNING: GROQ_API_KEY is not set. AI features (chat and comparison) will return errors until configured.');
} else {
    // Initialize Groq (key stays server-side only)
    try {
        groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
        console.log('Groq SDK initialized successfully');
    } catch (err) {
        console.error('Failed to initialize Groq SDK:', err.message);
    }
}

const MODEL = 'llama-3.1-8b-instant';

// =====================================================
// SECURITY: Rate limiters
// =====================================================

// General API rate limit: 30 requests per minute per IP
const generalLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 30,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many requests. Please wait a moment and try again.' }
});

// Chat rate limit: 10 AI requests per minute per IP (protects Groq quota)
const chatLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 10,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many AI requests. Please wait a minute before asking another question.' }
});

// Upload rate limit: 5 uploads per minute per IP
const uploadLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 5,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many uploads. Please wait a minute before uploading again.' }
});

const comparisonLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 3,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many comparison requests. Please wait a minute.' }
});

// =====================================================
// SECURITY: Input validation constants
// =====================================================
const MAX_DOCUMENT_SIZE = 500000;   // 500KB of text
const MAX_QUESTION_LENGTH = 1000;   // 1000 characters per question
const MIN_QUESTION_LENGTH = 2;      // At least 2 chars
const MAX_SESSION_ID_LENGTH = 50;   // UUID length check

// =====================================================
// SECURITY: Sanitize text input (strip dangerous content)
// =====================================================
function sanitizeText(text) {
    if (typeof text !== 'string') return '';
    // Remove null bytes and control characters (except newline/tab)
    return text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
}

function validateSessionId(id) {
    if (typeof id !== 'string') return false;
    if (id.length > MAX_SESSION_ID_LENGTH) return false;
    // UUID format check
    return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use('/api', generalLimiter);

// File upload config with stricter limits
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 5 * 1024 * 1024 }, // 5MB file limit
    fileFilter: (req, file, cb) => {
        // Only allow PDF and text files
        const allowedTypes = ['application/pdf', 'text/plain'];
        const allowedExts = ['.pdf', '.txt'];
        const ext = file.originalname.toLowerCase().slice(file.originalname.lastIndexOf('.'));
        if (allowedTypes.includes(file.mimetype) || allowedExts.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error('Only PDF and TXT files are allowed'));
        }
    }
});

// Session storage for document context
const sessions = new Map();

// =====================================================
// HELPER: Truncate text to stay within token limits
// =====================================================
function truncateForPrompt(text, maxChars = 16000) {
    if (text.length <= maxChars) return text;
    return text.substring(0, maxChars) + '\n\n[... Document truncated. Full text not shown.]';
}

// =====================================================
// API: Health check
// =====================================================
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', model: MODEL });
});

// =====================================================
// API: Upload document (creates session for Q&A)
// =====================================================
app.post('/api/upload', uploadLimiter, upload.single('file'), async (req, res) => {
    try {
        let documentText = '';

        if (req.file) {
            if (req.file.mimetype === 'application/pdf') {
                const pdfData = await pdfParse(req.file.buffer);
                documentText = pdfData.text;
            } else {
                documentText = req.file.buffer.toString('utf-8');
            }
        } else if (req.body.text || req.body.documentText) {
            documentText = req.body.text || req.body.documentText;
        } else {
            return res.status(400).json({ error: 'No document provided' });
        }

        // Sanitize and validate document text
        documentText = sanitizeText(documentText);

        if (!documentText.trim()) {
            return res.status(400).json({ error: 'Document is empty' });
        }

        if (documentText.length > MAX_DOCUMENT_SIZE) {
            return res.status(413).json({
                error: `Document is too large (${Math.round(documentText.length / 1000)}KB). Maximum allowed is ${MAX_DOCUMENT_SIZE / 1000}KB.`
            });
        }

        // Create session
        const sessionId = uuidv4();
        sessions.set(sessionId, {
            documentText,
            createdAt: Date.now()
        });

        console.log(`Session created: ${sessionId} (${documentText.length} chars)`);

        res.json({ success: true, sessionId });

    } catch (error) {
        console.error('Upload error:', error.message);
        if (error.message === 'Only PDF and TXT files are allowed') {
            return res.status(400).json({ error: error.message });
        }
        res.status(500).json({ error: 'Upload failed. Please try again.' });
    }
});

// =====================================================
// API: Q&A Chat (Groq-powered)
// =====================================================
app.post('/api/chat', chatLimiter, async (req, res) => {
    try {
        const { sessionId, question } = req.body;

        // Validate session ID
        if (!sessionId || !validateSessionId(sessionId)) {
            return res.status(400).json({ error: 'Invalid session.' });
        }

        // Validate question
        if (!question || typeof question !== 'string') {
            return res.status(400).json({ error: 'Please enter a question.' });
        }

        const sanitizedQuestion = sanitizeText(question).trim();

        if (sanitizedQuestion.length < MIN_QUESTION_LENGTH) {
            return res.status(400).json({ error: 'Question is too short.' });
        }

        if (sanitizedQuestion.length > MAX_QUESTION_LENGTH) {
            return res.status(400).json({
                error: `Question is too long (${sanitizedQuestion.length} chars). Maximum is ${MAX_QUESTION_LENGTH} characters.`
            });
        }

        const session = sessions.get(sessionId);
        if (!session) {
            return res.status(404).json({ error: 'Session expired. Please upload document again.' });
        }

        // Truncate document to fit token limits
        const truncatedDoc = truncateForPrompt(session.documentText);

        // Build Q&A prompt
        const qaPrompt = prompts.QA_PROMPT
            .replace('{document}', truncatedDoc)
            .replace('{question}', sanitizedQuestion);

        console.log(`Chat Q: "${sanitizedQuestion.substring(0, 60)}..." (doc: ${truncatedDoc.length} chars)`);

        // Check if Groq is initialized
        if (!groq) {
            return res.status(500).json({
                error: 'AI service is not configured. Please ensure GROQ_API_KEY is set in your environment variables.',
                details: 'Missing GROQ_API_KEY'
            });
        }

        // Call Groq
        const chatCompletion = await groq.chat.completions.create({
            messages: [
                { role: 'system', content: prompts.SYSTEM_CONTEXT + '\nAlways respond with valid JSON only.' },
                { role: 'user', content: qaPrompt }
            ],
            model: MODEL,
            temperature: 0.3,
            max_tokens: 1024,
            response_format: { type: 'json_object' }
        });

        const text = chatCompletion.choices[0]?.message?.content || '{}';
        const result = JSON.parse(text);

        console.log(`Chat A: "${(result.answer || '').substring(0, 80)}..."`);

        res.json({ success: true, ...result });

    } catch (error) {
        console.error('Chat error:', error.message);

        if (error.status === 429 || error.code === 'rate_limit_exceeded') {
            return res.status(429).json({
                error: 'AI is busy. Please wait a moment and try again.'
            });
        }
        if (error.status === 413) {
            return res.status(413).json({
                error: 'Document is too large for AI processing. Try with a shorter document.'
            });
        }

        res.status(500).json({ error: 'Chat failed. Please try again.' });
    }
});

// =====================================================
// Cleanup old sessions (every hour)
// =====================================================
setInterval(() => {
    const oneHourAgo = Date.now() - (60 * 60 * 1000);
    let cleaned = 0;
    for (const [id, session] of sessions.entries()) {
        if (session.createdAt < oneHourAgo) {
            sessions.delete(id);
            cleaned++;
        }
    }
    if (cleaned > 0) console.log(`Cleaned up ${cleaned} expired sessions`);
}, 60 * 60 * 1000);

// =====================================================
// API: Compare two documents
// =====================================================
app.post('/api/compare', generalLimiter, comparisonLimiter, async (req, res) => {
    try {
        const { doc1, doc2 } = req.body;

        if (!doc1 || !doc2) {
            return res.status(400).json({ error: 'Both documents are required for comparison.' });
        }

        const prompt = `
        You are a legal expert. Compare the following two versions of a document (or two related contracts). 
        Provide a concise, high-level analysis of the most important differences in plain English.
        Focus on:
        1. Financial changes (fees, payments, penalties)
        2. Scope of work or obligations
        3. Risks and liabilities
        4. Termination or duration changes

        Format your response as HTML (no <html> or <body> tags) using <h4> for section titles and <div class="diff-point"><strong>Topic:</strong> Description</div> for each point.

        DOCUMENT 1 (Current):
        ${doc1.substring(0, 10000)}

        DOCUMENT 2:
        ${doc2.substring(0, 10000)}
        `;

        // Check if Groq is initialized
        if (!groq) {
            return res.status(500).json({
                error: 'AI service is not configured. Please ensure GROQ_API_KEY is set in your environment variables.',
                details: 'Missing GROQ_API_KEY'
            });
        }

        const chatCompletion = await groq.chat.completions.create({
            messages: [{ role: 'user', content: prompt }],
            model: "llama-3.3-70b-versatile",
            temperature: 0.2,
            max_tokens: 2000,
        });

        const analysis = chatCompletion.choices[0]?.message?.content || 'No significant differences found.';
        res.json({ analysis });

    } catch (err) {
        console.error('Comparison API error:', err);
        res.status(500).json({ error: 'Failed to generate comparison. Please try again later.' });
    }
});

// =====================================================
// HELPERS
// =====================================================
const startServer = () => {
    app.listen(PORT, () => {
        console.log(`
╔═══════════════════════════════════════════════════════╗
║   LEGAL DOCUMENT DECODER                              ║
║   Analysis: Client-side (instant)                     ║
║   Chat Q&A: Groq AI (${MODEL})              ║
╠═══════════════════════════════════════════════════════╣
║   http://localhost:${PORT}                               ║
║   Security: Rate limiting + input validation ✓        ║
╚═══════════════════════════════════════════════════════╝
        `);
    });
};

// Only listen when running locally or in standard environments (not required for Vercel/Serverless)
if (require.main === module) {
    startServer();
}

module.exports = app;
