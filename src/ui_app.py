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
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline, create_simple_pipeline, RAGResponse
from vector_store_utils import ComplaintVectorStore


class ChatInterface:
    """Main chat interface class for the RAG system."""
    
    def __init__(self):
        self.rag_pipeline = None
        self.vector_store = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
    
    def initialize_rag_system(self) -> bool:
        """Initialize the RAG pipeline and vector store."""
        try:
            vector_store_dir = "../vector_store"
            if not os.path.exists(vector_store_dir):
                st.error("❌ Vector store not found! Please run Task 2 notebook first.")
                return False
            
            with st.spinner("Loading RAG system..."):
                self.rag_pipeline = create_simple_pipeline(vector_store_dir)
                self.vector_store = ComplaintVectorStore(vector_store_dir)
                self.vector_store.load()
                
                stats = self.vector_store.get_stats()
                st.success(f"✅ RAG System loaded: {stats['total_chunks']} chunks available")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            return False
    
    def render_header(self):
        """Render the application header."""
        st.set_page_config(
            page_title="Financial Complaint Analysis RAG System",
            page_icon="🏦",
            layout="wide"
        )

        st.title("🏦 Financial Complaint Analysis RAG System")
        st.markdown("**AI-powered analysis of consumer financial complaints**")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        # RAG Pipeline Settings
        top_k = st.sidebar.slider("Sources to retrieve", 1, 10, 5)
        show_sources = st.sidebar.checkbox("Show source details", value=True)
        show_metadata = st.sidebar.checkbox("Show source metadata", value=True)
        
        # Example questions
        st.sidebar.subheader("💡 Example Questions")
        examples = [
            "What are common credit card billing issues?",
            "How do customers resolve billing disputes?",
            "What are main security concerns?",
            "Which products have most complaints?"
        ]
        
        for example in examples:
            if st.sidebar.button(example, key=f"example_{hash(example)}"):
                st.session_state.current_question = example
                st.rerun()
        
        return {'top_k': top_k, 'show_sources': show_sources, 'show_metadata': show_metadata}
    
    def render_chat_interface(self, config):
        """Render the main chat interface."""
        st.subheader("💬 Ask Your Question")
        
        # Question input
        question = st.text_area(
            "Type your question:",
            value=st.session_state.current_question,
            height=100,
            placeholder="e.g., What are common credit card billing issues?"
        )
        
        # Submit and Clear buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_question = ""
                st.rerun()
        
        # Process question if submitted
        if submit_button and question.strip():
            self.process_question(question.strip(), config)
            st.session_state.current_question = ""
    
    def process_question(self, question: str, config):
        """Process a user question and generate response."""
        if not self.rag_pipeline:
            st.error("❌ RAG system not initialized.")
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
        with st.chat_message("user"):
            st.write(f"**Question:** {question}")
        
        # Process with RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    start_time = time.time()
                    response = self.rag_pipeline.run(question, k=config['top_k'])
                    processing_time = time.time() - start_time
                    
                    chat_entry['response'] = response
                    chat_entry['processing_time'] = processing_time
                    
                    self.display_response(response, config, processing_time)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    def display_response(self, response: RAGResponse, config, processing_time: float):
        """Display the RAG response."""
        st.markdown("**🤖 AI Response:**")
        st.markdown(response.answer)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("🎯 Confidence", f"{response.confidence_score:.3f}")
        with col3:
            st.metric("📚 Sources", len(response.retrieved_sources))
        
        # Display sources
        if config['show_sources'] and response.retrieved_sources:
            self.display_sources(response.retrieved_sources, config)
    
    def display_sources(self, sources, config):
        """Display the retrieved sources."""
        st.markdown("---")
        st.subheader("📚 Retrieved Sources")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i} - {source.metadata.get('category', 'Unknown')}", expanded=i <= 2):
                if config['show_metadata']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Product:** {source.metadata.get('product', 'Unknown')}")
                    with col2:
                        st.markdown(f"**Category:** {source.metadata.get('category', 'Unknown')}")
                    with col3:
                        st.markdown(f"**Score:** {source.score:.3f}")
                
                st.markdown("**Text:**")
                source_text = source.text[:500] + "..." if len(source.text) > 500 else source.text
                st.text_area(f"Source {i}", value=source_text, height=150, disabled=True)
    
    def render_chat_history(self):
        """Render the chat history."""
        if not st.session_state.chat_history:
            return
        
        st.markdown("---")
        st.subheader("📜 Chat History")
        
        for i, entry in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {entry['question'][:50]}...", expanded=False):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Time:** {entry['timestamp'].strftime('%H:%M:%S')}")
                
                if entry['response']:
                    response = entry['response']
                    st.markdown(f"**Answer:** {response.answer[:200]}...")
                    st.metric("Processing Time", f"{entry['processing_time']:.2f}s")
                    st.metric("Confidence", f"{response.confidence_score:.3f}")
    
    def run(self):
        """Run the Streamlit application."""
        if not self.initialize_rag_system():
            st.error("❌ Failed to initialize RAG system.")
            return
        
        self.render_header()
        config = self.render_sidebar()
        self.render_chat_interface(config)
        self.render_chat_history()
        
        st.markdown("---")
        st.markdown("**Built with ❤️ using Streamlit, Sentence Transformers, and FAISS**")


def main():
    """Main entry point for the Streamlit application. """
    chat_interface = ChatInterface()
    chat_interface.run()


if __name__ == "__main__":
    main()