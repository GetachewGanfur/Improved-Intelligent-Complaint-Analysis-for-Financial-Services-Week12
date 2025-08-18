"""
Interactive Chat Interface for Financial Complaint Analysis RAG System

This module provides a Streamlit-based web interface that allows users to interact
with the RAG system through an intuitive chat interface.
"""

import streamlit as st
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline, create_simple_pipeline, RAGResponse
from vector_store_utils import ComplaintVectorStore


class ChatInterface:
    """Main chat interface class for the RAG system."""
    
    def __init__(self):
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.vector_store: Optional[ComplaintVectorStore] = None
        self.initialize_session_state()
    
    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        session_defaults = {
            'chat_history': [],
            'current_question': "",
            'system_initialized': False
        }
        
        for key, value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def initialize_rag_system(self) -> bool:
        """Initialize the RAG pipeline and vector store."""
        if st.session_state.get('system_initialized', False):
            return True
            
        try:
            vector_store_dir = "../vector_store"
            if not os.path.exists(vector_store_dir):
                st.error("❌ Vector store not found! Please run the data preparation notebook first.")
                return False
            
            with st.spinner("Loading RAG system. This may take a minute..."):
                start_time = time.time()
                self.rag_pipeline = create_simple_pipeline(vector_store_dir)
                self.vector_store = ComplaintVectorStore(vector_store_dir)
                self.vector_store.load()
                
                stats = self.vector_store.get_stats()
                load_time = time.time() - start_time
                
                st.success(f"""
                ✅ RAG System Successfully Loaded:
                - {stats['total_chunks']} complaint chunks available
                - Loaded in {load_time:.2f} seconds
                """)
                
                st.session_state.system_initialized = True
                return True
            
        except Exception as e:
            st.error(f"""
            ❌ Failed to initialize RAG system:
            {str(e)}
            
            Please check that:
            1. The vector store exists at {vector_store_dir}
            2. You have all required dependencies installed
            """)
            return False
    
    def render_header(self) -> None:
        """Render the application header."""
        st.set_page_config(
            page_title="Financial Complaint Analysis RAG System",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🏦 Financial Complaint Analysis RAG System")
        st.markdown("""
        **AI-powered analysis of consumer financial complaints**  
        Ask questions about consumer financial complaints and get insights powered by retrieval-augmented generation.
        """)
        st.markdown("---")
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render the sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        # System Status
        st.sidebar.subheader("System Status")
        if st.session_state.system_initialized:
            st.sidebar.success("✅ System Ready")
        else:
            st.sidebar.warning("⚠️ Initializing...")
        
        # RAG Pipeline Settings
        st.sidebar.subheader("Search Settings")
        config = {
            'top_k': st.sidebar.slider(
                "Number of sources to retrieve",
                min_value=1,
                max_value=10,
                value=5,
                help="More sources may provide better answers but will be slower"
            ),
            'show_sources': st.sidebar.checkbox(
                "Show source details",
                value=True,
                help="Display the retrieved complaint segments"
            ),
            'show_metadata': st.sidebar.checkbox(
                "Show source metadata",
                value=True,
                help="Display product/category information for sources"
            )
        }
        
        # Example questions
        st.sidebar.subheader("💡 Example Questions")
        examples = [
            "What are the most common credit card billing issues?",
            "How do customers typically resolve mortgage disputes?",
            "What security concerns are mentioned in checking account complaints?",
            "Which financial products have the most complaints about fraud?"
        ]
        
        for example in examples:
            if st.sidebar.button(
                example,
                key=f"example_{hash(example)}",
                help="Click to try this example question",
                use_container_width=True
            ):
                st.session_state.current_question = example
                st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **About this system:**  
        This RAG system analyzes consumer complaints from the CFPB database.
        """)
        
        return config
    
    def render_chat_interface(self, config: Dict[str, Any]) -> None:
        """Render the main chat interface."""
        st.subheader("💬 Ask Your Question")
        
        # Question input
        question = st.text_area(
            "Type your question about consumer financial complaints:",
            value=st.session_state.current_question,
            height=100,
            placeholder="e.g., What are common credit card billing issues?",
            label_visibility="collapsed"
        )
        
        # Action buttons
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button(
                "🗑️ Clear Chat",
                help="Clear all conversation history",
                use_container_width=True
            ):
                st.session_state.chat_history = []
                st.session_state.current_question = ""
                st.rerun()
        with col2:
            submit_button = st.button(
                "🚀 Ask Question",
                type="primary",
                disabled=not st.session_state.system_initialized,
                help="Ask your question to the RAG system" if st.session_state.system_initialized else "System still initializing...",
                use_container_width=True
            )
        
        # Process question if submitted
        if submit_button and question.strip():
            self.process_question(question.strip(), config)
            st.session_state.current_question = ""
            st.rerun()
    
    def process_question(self, question: str, config: Dict[str, Any]) -> None:
        """Process a user question and generate response."""
        if not self.rag_pipeline or not st.session_state.system_initialized:
            st.error("❌ RAG system not properly initialized.")
            return
        
        # Add to chat history
        chat_entry = {
            'timestamp': datetime.now(),
            'question': question,
            'response': None,
            'processing_time': None
        }
        st.session_state.chat_history.append(chat_entry)
        
        # Display question
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**You:** {question}")
        
        # Process with RAG pipeline
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing complaints database..."):
                try:
                    start_time = time.time()
                    response = self.rag_pipeline.run(question, k=config['top_k'])
                    processing_time = time.time() - start_time
                    
                    chat_entry['response'] = response
                    chat_entry['processing_time'] = processing_time
                    
                    self.display_response(response, config, processing_time)
                except Exception as e:
                    st.error(f"""
                    ❌ Error processing your question:
                    {str(e)}
                    
                    Please try again with a different question or check back later.
                    """)
    
    def display_response(self, response: RAGResponse, config: Dict[str, Any], processing_time: float) -> None:
        """Display the RAG response."""
        # Main answer
        st.markdown("**Response:**")
        st.markdown(response.answer)
        
        # Performance metrics
        with st.expander("⚡ Performance Metrics", expanded=False):
            cols = st.columns(3)
            cols[0].metric("Processing Time", f"{processing_time:.2f}s")
            cols[1].metric("Confidence Score", f"{response.confidence_score:.3f}")
            cols[2].metric("Sources Retrieved", len(response.retrieved_sources))
        
        # Display sources if enabled
        if config['show_sources'] and response.retrieved_sources:
            self.display_sources(response.retrieved_sources, config)
    
    def display_sources(self, sources: list, config: Dict[str, Any]) -> None:
        """Display the retrieved sources."""
        st.markdown("---")
        st.subheader("📚 Retrieved Complaint Segments")
        st.caption("These are the most relevant complaint segments from the database")
        
        for i, source in enumerate(sources, 1):
            with st.expander(
                f"Segment {i}: {source.metadata.get('product', 'Unknown')} - {source.metadata.get('category', 'Unknown')}",
                expanded=i == 1  # Only expand first by default
            ):
                if config['show_metadata']:
                    metadata_cols = st.columns(4)
                    metadata_cols[0].metric("Relevance", f"{source.score:.3f}")
                    metadata_cols[1].markdown(f"**Product:**\n{source.metadata.get('product', 'N/A')}")
                    metadata_cols[2].markdown(f"**Issue:**\n{source.metadata.get('category', 'N/A')}")
                    metadata_cols[3].markdown(f"**Sub-issue:**\n{source.metadata.get('subcategory', 'N/A')}")
                
                st.markdown("**Complaint Text:**")
                st.text_area(
                    f"Source_{i}_text",
                    value=source.text,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
    
    def render_chat_history(self) -> None:
        """Render the chat history sidebar."""
        if not st.session_state.chat_history:
            return
            
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Conversation History")
        
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            if entry['response']:
                summary = f"Q{i+1}: {entry['question'][:30]}..."
                if st.sidebar.button(
                    summary,
                    key=f"history_{i}",
                    help=f"Asked at {entry['timestamp'].strftime('%H:%M')}",
                    use_container_width=True
                ):
                    st.session_state.current_question = entry['question']
                    st.rerun()
    
    def run(self) -> None:
        """Run the Streamlit application."""
        self.render_header()
        
        if not self.initialize_rag_system():
            st.error("Please fix the initialization errors to continue.")
            return
        
        config = self.render_sidebar()
        self.render_chat_interface(config)
        self.render_chat_history()
        
        st.markdown("---")
        st.caption("""
        **About this system:**  
        This Retrieval-Augmented Generation (RAG) system analyzes consumer complaints 
        from the CFPB database using FAISS for vector search and transformer models 
        for question answering.
        """)


def main():
    """Main entry point for the Streamlit application."""
    chat_interface = ChatInterface()
    chat_interface.run()


if __name__ == "__main__":
    main()