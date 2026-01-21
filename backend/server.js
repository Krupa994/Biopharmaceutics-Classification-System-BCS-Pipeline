// backend/server.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(express.json());
app.use(cors({ origin: process.env.FRONTEND_URL || 'http://localhost:5173' }));

// ----------------------------
// API Route
// ----------------------------
app.post('/api/predict', async (req, res) => {
    const smiles = req.body?.smiles;
    if (!smiles) return res.status(400).json({ error: 'smiles field required' });

    try {
        // Call the FastAPI ML service
        const response = await axios.post('http://localhost:8000/predict', { smiles });
        const data = response.data;

        // Forward the ML service response to the frontend
        return res.json(data);

    } catch (err) {
        console.error('Error calling ML service:', err.message);
        return res.status(500).json({ error: 'ML service unreachable or returned error' });
    }
});

// ----------------------------
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend listening on ${PORT}`));
