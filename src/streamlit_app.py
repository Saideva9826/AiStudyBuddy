import streamlit as st
import os
import tempfile
from rag_system import RAGSystem

# Page configuration
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_system():
    """Initialize the RAG system with API key."""
    api_key = st.session_state.get("openai_api_key", "")
    if api_key:
        st.session_state.rag_system = RAGSystem(openai_api_key=api_key)
    else:
        st.session_state.rag_system = RAGSystem()  # Mock mode
    return st.session_state.rag_system

def process_uploaded_files(uploaded_files):
    """Process uploaded files and create vector store."""
    if not uploaded_files:
        return False
    
    # Save uploaded files temporarily
    temp_files = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_files.append(tmp_file.name)
    
    # Initialize RAG system if not already done
    if not st.session_state.rag_system:
        initialize_rag_system()
    
    # Load documents
    try:
        vectorstore_type = st.session_state.get("vectorstore_type", "faiss")
        success = st.session_state.rag_system.load_documents(temp_files, vectorstore_type=vectorstore_type)
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        return success
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
        return False

# Main app
def main():
    st.title("📚 AI Study Buddy")
    st.markdown("*Your intelligent companion for studying and learning*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            help="Enter your OpenAI API key for full functionality. Leave empty for demo mode."
        )
        if api_key:
            st.session_state.openai_api_key = api_key
        
        # Vector store selection
        vectorstore_type = st.selectbox(
            "Vector Store",
            ["faiss", "chroma"],
            help="Choose the vector database for storing document embeddings."
        )
        st.session_state.vectorstore_type = vectorstore_type
        
        st.divider()
        
        # File upload
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose your study materials",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files containing your study materials."
        )
        
        if uploaded_files and st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                success = process_uploaded_files(uploaded_files)
                if success:
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
                    st.rerun()
                else:
                    st.error("❌ Failed to process documents.")
        
        # Document status
        if st.session_state.documents_loaded:
            st.success("📚 Documents loaded and ready!")
        else:
            st.info("📝 Upload documents to get started.")
    
    # Main content area
    if not st.session_state.documents_loaded:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ## Welcome to AI Study Buddy! 🎓
            
            Get started by:
            1. **Upload your study materials** (PDF, DOCX, TXT) using the sidebar
            2. **Optionally add your OpenAI API key** for enhanced responses
            3. **Start asking questions** about your materials
            
            ### Features:
            - 💬 **Q&A**: Ask questions about your study materials
            - 📝 **Summarization**: Get concise summaries of your documents
            - 🃏 **Flashcards**: Generate study flashcards automatically
            - 🔍 **Context-aware**: Responses based on your specific materials
            """)
    else:
        # Main interface with tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Summary", "🃏 Flashcards"])
        
        with tab1:
            st.header("Chat with your Study Materials")
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and "context" in message:
                        with st.expander("📖 View Context"):
                            for i, context in enumerate(message["context"], 1):
                                st.text_area(f"Context {i}", context, height=100, disabled=True)
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your study materials..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = st.session_state.rag_system.answer_question(prompt)
                        st.write(result["response"])
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": result["response"],
                            "context": result["context_docs"]
                        })
                        
                        # Show context
                        if result["context_docs"]:
                            with st.expander("📖 View Context"):
                                for i, context in enumerate(result["context_docs"], 1):
                                    st.text_area(f"Context {i}", context, height=100, disabled=True)
        
        with tab2:
            st.header("Document Summary")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                max_length = st.slider("Summary Length (words)", 50, 500, 200)
            with col2:
                if st.button("Generate Summary", type="primary"):
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.rag_system.summarize_documents(max_length)
                        st.session_state.summary = summary
            
            if hasattr(st.session_state, 'summary'):
                st.markdown("### 📄 Summary")
                st.write(st.session_state.summary)
        
        with tab3:
            st.header("Flashcards Generator")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                topic = st.text_input("Topic (optional)", placeholder="e.g., machine learning, history")
            with col2:
                num_cards = st.number_input("Number of Cards", 1, 10, 5)
            with col3:
                if st.button("Generate Flashcards", type="primary"):
                    with st.spinner("Creating flashcards..."):
                        flashcards = st.session_state.rag_system.generate_flashcards(topic, num_cards)
                        st.session_state.flashcards = flashcards
            
            if hasattr(st.session_state, 'flashcards'):
                st.markdown("### 🃏 Your Flashcards")
                for i, card in enumerate(st.session_state.flashcards, 1):
                    with st.expander(f"Card {i}: {card['question'][:50]}..."):
                        st.markdown(f"**Question:** {card['question']}")
                        st.markdown(f"**Answer:** {card['answer']}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Made with ❤️ using Streamlit, LangChain, and OpenAI
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

