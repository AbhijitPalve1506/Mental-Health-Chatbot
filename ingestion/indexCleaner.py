import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 🧹 Clear vectors (from "cbt" namespace if that’s where you ingested)
try:
    index.delete(deleteAll=True, namespace="cbt")
    print(f"🧹 Cleared all vectors from namespace 'default' in index '{INDEX_NAME}'")
except Exception as e:
    print(f"⚠️ Could not delete vectors: {e}")

# Optional: check stats after cleanup
print("📊 Index stats:", index.describe_index_stats())