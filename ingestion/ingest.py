import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "mental-health-chatbot")

def clean_text(text: str) -> str:
    """Basic preprocessing: remove newlines, extra spaces, and non-ASCII chars."""
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    text = text.strip()
    return text

def ingest_pdf(pdf_path: str, namespace: str, embeddings, pc):
    """Ingest a single PDF into Pinecone under its own namespace."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found at {pdf_path}")
    
    print(f"📖 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages from {os.path.basename(pdf_path)}")

    # Clean text
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = splitter.split_documents(docs)
    print(f"✂️ Split {os.path.basename(pdf_path)} into {len(texts)} chunks")

    # Assign unique IDs per chunk
    for i, t in enumerate(texts):
        t.metadata["id"] = f"{namespace}-{i}"

    # Store in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    print(f"✅ Ingested {len(texts)} chunks from {os.path.basename(pdf_path)} into namespace `{namespace}`")

def main():
    print("🚀 Starting multi-PDF ingestion...")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        raise ValueError(f"❌ Index '{INDEX_NAME}' does not exist in Pinecone. Please create it first.")

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    # Define PDFs and namespaces
    pdfs = {
        "wyt": "./data/You Become What You Think.pdf",
        "mhb": "./data/Mental Health Care Book.pdf"
    }

    # Ingest each PDF
    for namespace, pdf_path in pdfs.items():
        ingest_pdf(pdf_path, namespace, embeddings, pc)

    # Show stats
    index = pc.Index(INDEX_NAME)
    print("📊 Final Index Stats:", index.describe_index_stats())

if __name__ == "__main__":
    main()