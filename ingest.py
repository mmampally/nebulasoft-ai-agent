import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    # Loading text
    doc_path = "knowledge_base/nebula_manual.txt"
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_path}")
    
    with open(doc_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"Loaded document: {len(raw_text)} characters")
    
    # Splitting into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(raw_text)
    print(f"Created {len(chunks)} chunks")
    
    # Create documents with metadata (for citation)
    from langchain_core.documents import Document
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": "nebula_manual.txt", "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Loading embeddings model (LangChain-compatible wrapper)
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Building Chroma vector DB (easier than FAISS for this use case)
    db_path = "./chroma_db"
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    print(f"✅ Vector DB saved to {db_path}")
    print(f"✅ Total chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    main()