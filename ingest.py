import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ✅ CENTRALIZED CONFIG
NEBULA_DOC_PATH = "knowledge_base/nebula_manual.txt"
PERSIST_DIR = "./chroma_nebula"   # Renamed for clarity


def main():
    # ✅ 1. Load NebulaSoft Manual
    if not os.path.exists(NEBULA_DOC_PATH):
        raise FileNotFoundError(f"❌ Document not found: {NEBULA_DOC_PATH}")
    
    with open(NEBULA_DOC_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"✅ Loaded NebulaSoft manual: {len(raw_text)} characters")
    
    # ✅ 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(raw_text)
    print(f"✅ Created {len(chunks)} chunks")
    
    # ✅ 3. Create documents with proper metadata (for citation later)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": "nebula_manual.txt",
                "chunk_id": i,
                "kb": "nebula"
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # ✅ 4. Load Embeddings Model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # ✅ 5. Reset + Rebuild Vector DB (Safe Local Behavior)
    if os.path.exists(PERSIST_DIR):
        print(f"⚠️ Existing vector DB found at {PERSIST_DIR}. Rebuilding it...")
    
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="nebula_kb"
    )
    
    db.persist()
    
    print(f"✅ NebulaSoft vector DB saved to: {PERSIST_DIR}")
    print(f"✅ Total NebulaSoft chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
