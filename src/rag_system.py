import os
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from document_processor import process_document
from embedding_vectorstore import EmbeddingVectorStoreManager

class RAGSystem:
    def __init__(self, openai_api_key=None, model_name="gpt-3.5-turbo", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # For demo purposes, we'll use a mock client
            self.client = None
            print("Warning: No OpenAI API key provided. Using mock responses.")
        
        self.model_name = model_name
        self.embedding_manager = EmbeddingVectorStoreManager(model_name=embedding_model)
        self.vectorstore = None

    def load_documents(self, file_paths, vectorstore_type="faiss", db_name="study_buddy_db"):
        """Load documents and create vector store."""
        all_chunks = []
        
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            chunks = process_document(file_path)
            all_chunks.extend(chunks)
        
        print(f"Total chunks: {len(all_chunks)}")
        
        if vectorstore_type.lower() == "faiss":
            self.vectorstore = self.embedding_manager.create_faiss_vectorstore(all_chunks, db_name)
        elif vectorstore_type.lower() == "chroma":
            self.vectorstore = self.embedding_manager.create_chroma_vectorstore(all_chunks, db_name)
        else:
            raise ValueError("Unsupported vectorstore type. Use 'faiss' or 'chroma'.")
        
        return self.vectorstore is not None

    def retrieve_context(self, query, k=3):
        """Retrieve relevant context from vector store."""
        if not self.vectorstore:
            return []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def generate_response(self, query, context_docs):
        """Generate response using LLM with retrieved context."""
        if not self.client:
            # Mock response for demo
            return f"Mock response for query: '{query}' based on {len(context_docs)} context documents."
        
        # Prepare context
        context = "\n\n".join(context_docs)
        
        # Create prompt
        prompt = f"""You are an AI Study Buddy helping students understand their study materials. 
        Use the following context to answer the student's question. If the answer is not in the context, 
        say so and provide general guidance.

        Context:
        {context}

        Question: {query}

        Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI Study Buddy for students."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_question(self, query, k=3):
        """Complete RAG pipeline: retrieve context and generate answer."""
        context_docs = self.retrieve_context(query, k)
        response = self.generate_response(query, context_docs)
        
        return {
            "query": query,
            "context_docs": context_docs,
            "response": response
        }

    def summarize_documents(self, max_length=200):
        """Generate a summary of all loaded documents."""
        if not self.vectorstore:
            return "No documents loaded."
        
        # Get a sample of documents for summarization
        sample_docs = self.retrieve_context("overview summary", k=5)
        
        if not self.client:
            return f"Mock summary of {len(sample_docs)} document chunks."
        
        context = "\n\n".join(sample_docs)
        prompt = f"""Summarize the following study material in {max_length} words or less:

        {context}

        Summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI that creates concise summaries of study materials."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_flashcards(self, topic=None, num_cards=5):
        """Generate flashcards from the study material."""
        if not self.vectorstore:
            return []
        
        # Retrieve relevant content
        if topic:
            context_docs = self.retrieve_context(topic, k=3)
        else:
            context_docs = self.retrieve_context("key concepts important facts", k=5)
        
        if not self.client:
            # Mock flashcards
            return [
                {"question": f"Mock question {i+1} about {topic or 'study material'}", 
                 "answer": f"Mock answer {i+1}"}
                for i in range(num_cards)
            ]
        
        context = "\n\n".join(context_docs)
        prompt = f"""Create {num_cards} flashcards from the following study material. 
        Format each flashcard as "Q: [question]" followed by "A: [answer]".

        Study Material:
        {context}

        Flashcards:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI that creates educational flashcards."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            # Parse flashcards from response
            flashcards = []
            lines = response.choices[0].message.content.split('\n')
            current_question = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    current_question = line[2:].strip()
                elif line.startswith('A:') and current_question:
                    answer = line[2:].strip()
                    flashcards.append({"question": current_question, "answer": answer})
                    current_question = None
            
            return flashcards[:num_cards]
        except Exception as e:
            return [{"question": f"Error generating flashcards: {str(e)}", "answer": "Please try again."}]

if __name__ == "__main__":
    # Demo usage
    rag_system = RAGSystem()  # No API key for demo
    
    # Create a dummy study material file
    dummy_file = "study_material.txt"
    with open(dummy_file, "w") as f:
        f.write("""
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence (AI) that enables systems to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.
        
        Types of Machine Learning:
        1. Supervised Learning: Uses labeled training data to learn a mapping from inputs to outputs.
        2. Unsupervised Learning: Finds hidden patterns in data without labeled examples.
        3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties.
        
        Key Concepts:
        - Training Data: The dataset used to teach the machine learning algorithm.
        - Features: Individual measurable properties of observed phenomena.
        - Model: The mathematical representation of a real-world process.
        - Algorithm: The method used to build the model.
        
        Popular Algorithms:
        - Linear Regression: For predicting continuous values.
        - Decision Trees: For classification and regression tasks.
        - Neural Networks: For complex pattern recognition.
        - Support Vector Machines: For classification and regression.
        """)
    
    print("=== AI Study Buddy RAG System Demo ===")
    
    # Load documents
    success = rag_system.load_documents([dummy_file], vectorstore_type="faiss")
    if success:
        print("✓ Documents loaded successfully!")
        
        # Test Q&A
        print("\n--- Question Answering ---")
        result = rag_system.answer_question("What is machine learning?")
        print(f"Q: {result['query']}")
        print(f"A: {result['response']}")
        print(f"Context used: {len(result['context_docs'])} documents")
        
        # Test summarization
        print("\n--- Document Summary ---")
        summary = rag_system.summarize_documents()
        print(f"Summary: {summary}")
        
        # Test flashcard generation
        print("\n--- Flashcards ---")
        flashcards = rag_system.generate_flashcards(topic="machine learning", num_cards=3)
        for i, card in enumerate(flashcards, 1):
            print(f"Card {i}:")
            print(f"  Q: {card['question']}")
            print(f"  A: {card['answer']}")
    
    # Cleanup
    os.remove(dummy_file)
    print("\n✓ Demo completed and cleaned up!")

