# 🧠 Mental Health Chatbot 

A RAG-powered chatbot that provides information about Cognitive Behavioral Therapy (CBT) skills using a PDF workbook as the knowledge base.

## 🚀 Features

- **RAG Architecture**: Uses Pinecone vector database for semantic search
- **Gemini 2.0 Flash**: Powered by Google's latest LLM
- **FastAPI Backend**: RESTful API with automatic documentation
- **Streamlit Frontend**: Clean, user-friendly chat interface
- **PDF Processing**: Automatically ingests and chunks CBT workbook content

## 📋 Prerequisites

- Python 3.8+
- Pinecone account and API key
- Google AI API key
- Mental Health Care Book.pdf
- You Become What You think.pdf

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
INDEX_NAME=mental-health-chatbot
```

### 3. Data Ingestion

First, ensure your CBT workbook PDF is in the `data/` folder, then run:

```bash
python ingestion/ingest.py
```

This will:
- Load and chunk the PDF
- Generate embeddings using Google's text-embedding-004
- Store vectors in Pinecone

### 4. Start the Backend
this is a FastAPI app, the recommended way is to run it with uvicorn (ASGI server):
ex. if file name is api.py then run :
```bash
uvicorn api:app --reload
```

The FastAPI server will start at `https://mental-health-chatbot-backend-hi3p.onrender.com`

### 5. Start the Frontend

In a new terminal:

```bash
streamlit run frontend/app.py
```

The Streamlit app will open at `http://localhost:8501`

## 🔧 Troubleshooting

### Common Issues

1. **"Cannot connect to backend server"**
   - Ensure the FastAPI server is running 

2. **"No answer generated"**
   - Verify your Pinecone index contains data (`check on pinecone.io`)
   - Check that your API keys are correct
   - Ensure the PDF was ingested successfully

3. **Slow responses**
   - The first query may be slow due to cold start
   - Check your internet connection for API calls

### Health Checks

- **Backend Health**: `https://mental-health-chatbot-backend-hi3p.onrender.com`
- **API Docs**: `https://mental-health-chatbot-backend-hi3p.onrender.com/docs`

## 📁 Project Structure

```
Mental Health Chatbot/
├── backend/
│   └── api.py              # FastAPI backend with RAG logic
│   └── requirements.txt         # Python dependencies
├── data/
│   └── Mental Health Care Book.pdf
│   └── You Become What You think.pdf
├── frontend/
│   └── app.py              # Streamlit chat interface
├── ingestion/
│   └── ingest.py           # PDF processing and vector storage
│   └── indexCleaner.py           # To Clean the Vector storage to store new embeddigs
└── README.md               # This file
```

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- This chatbot is for educational purposes only, not medical advice

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 