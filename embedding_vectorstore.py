
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from document_processor import process_document # Assuming document_processor.py is in the same directory

class EmbeddingVectorStoreManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", persist_directory="./vector_db"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

    def create_faiss_vectorstore(self, documents, db_name="faiss_index"):
        print(f"Creating FAISS vector store: {db_name}")
        try:
            vectorstore = FAISS.from_texts(documents, self.embeddings)
            vectorstore.save_local(os.path.join(self.persist_directory, db_name))
            print(f"FAISS vector store saved to {os.path.join(self.persist_directory, db_name)}")
            return vectorstore
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")
            return None

    def load_faiss_vectorstore(self, db_name="faiss_index"):
        print(f"Loading FAISS vector store: {db_name}")
        try:
            vectorstore = FAISS.load_local(os.path.join(self.persist_directory, db_name), self.embeddings, allow_dangerous_deserialization=True)
            print(f"FAISS vector store loaded from {os.path.join(self.persist_directory, db_name)}")
            return vectorstore
        except Exception as e:
            print(f"Error loading FAISS vector store: {e}")
            return None

    def create_chroma_vectorstore(self, documents, collection_name="chroma_collection"):
        print(f"Creating Chroma vector store: {collection_name}")
        try:
            vectorstore = Chroma.from_texts(documents, self.embeddings, persist_directory=self.persist_directory, collection_name=collection_name)
            vectorstore.persist()
            print(f"Chroma vector store created in {self.persist_directory} with collection {collection_name}")
            return vectorstore
        except Exception as e:
            print(f"Error creating Chroma vector store: {e}")
            return None

    def load_chroma_vectorstore(self, collection_name="chroma_collection"):
        print(f"Loading Chroma vector store: {collection_name}")
        try:
            vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings, collection_name=collection_name)
            print(f"Chroma vector store loaded from {self.persist_directory} with collection {collection_name}")
            return vectorstore
        except Exception as e:
            print(f"Error loading Chroma vector store: {e}")
            return None

if __name__ == "__main__":
    manager = EmbeddingVectorStoreManager()

    # Example: Process a dummy text file and create vector stores
    dummy_text_path = "dummy_study_material.txt"
    with open(dummy_text_path, "w") as f:
        f.write("This is a test document for the AI Study Buddy. It contains information about Python programming. Python is a high-level, interpreted programming language. It is widely used for web development, data analysis, artificial intelligence, and more. The simplicity of its syntax makes it easy to learn.")
        f.write("\nAnother paragraph about machine learning. Machine learning is a subset of artificial intelligence that enables systems to learn from data. It involves algorithms that build a model from sample data, known as 'training data', in order to make predictions or decisions without being explicitly programmed to do so.")

    print("\n--- Processing dummy document ---")
    chunks = process_document(dummy_text_path, chunk_size=100, chunk_overlap=20)
    print(f"Generated {len(chunks)} chunks.")

    if chunks:
        # Create and test FAISS
        faiss_db = manager.create_faiss_vectorstore(chunks, db_name="my_faiss_db")
        if faiss_db:
            query = "What is Python?"
            docs = faiss_db.similarity_search(query)
            print(f"\nFAISS Search Results for \'{query}\':")
            for doc in docs:
                print(f"- {doc.page_content[:100]}...")

            # Load and test FAISS
            loaded_faiss_db = manager.load_faiss_vectorstore(db_name="my_faiss_db")
            if loaded_faiss_db:
                docs_loaded = loaded_faiss_db.similarity_search(query)
                print(f"FAISS Loaded Search Results for \'{query}\':")
                for doc in docs_loaded:
                    print(f"- {doc.page_content[:100]}...")

        # Create and test ChromaDB
        chroma_db = manager.create_chroma_vectorstore(chunks, collection_name="my_chroma_collection")
        if chroma_db:
            query = "What is machine learning?"
            docs = chroma_db.similarity_search(query)
            print(f"\nChromaDB Search Results for \'{query}\':")
            for doc in docs:
                print(f"- {doc.page_content[:100]}...")

            # Load and test ChromaDB
            loaded_chroma_db = manager.load_chroma_vectorstore(collection_name="my_chroma_collection")
            if loaded_chroma_db:
                docs_loaded = loaded_chroma_db.similarity_search(query)
                print(f"ChromaDB Loaded Search Results for \'{query}\':")
                for doc in docs_loaded:
                    print(f"- {doc.page_content[:100]}...")

    # Clean up dummy file and vector store directories
    os.remove(dummy_text_path)
    # For FAISS, you need to remove the directory
    import shutil
    if os.path.exists(os.path.join(manager.persist_directory, "my_faiss_db")):
        shutil.rmtree(os.path.join(manager.persist_directory, "my_faiss_db"))
    # For Chroma, the persist_directory itself contains the data
    if os.path.exists(manager.persist_directory):
        shutil.rmtree(manager.persist_directory)

    print("\n--- Cleaned up dummy files and vector stores ---")


