"""
Streamlit Chat Interface for Financial Complaint Analysis

This module provides a clean, modern Streamlit interface for the RAG system
with native chat components and intuitive user experience.
"""

import streamlit as st
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import warnings

# Suppress Streamlit warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
st.set_option('client.showErrorDetails', False)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from rag_pipeline import create_simple_pipeline, RAGResponse
    from chat_interface import ChatInterface, GradioResponseFormatter
    from vector_store_utils import ComplaintVectorStore
except ImportError as e:
    st.error(f"Error importing RAG components: {e}")
    st.stop()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "rag_pipeline" not in st.session_state:
        with st.spinner("🔧 Initializing RAG system..."):
            try:
                vector_store_dir = str(Path(__file__).parent.parent / "vector_store")
                st.session_state.rag_pipeline = create_simple_pipeline(vector_store_dir)
                st.session_state.chat_interface = ChatInterface(st.session_state.rag_pipeline)
                st.success("✅ RAG system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "session_start": time.time()
        }


def format_response_for_streamlit(response: RAGResponse) -> Dict:
    """Format RAG response for Streamlit display."""
    return {
        "answer": response.answer,
        "sources": response.retrieved_sources,
        "confidence": response.confidence_score,
        "processing_time": response.processing_time,
        "query": response.query
    }


def display_sources(sources: List, confidence: float):
    """Display source information in an expandable section."""
    if not sources:
        st.info("No sources available for this response.")
        return
    
    # Confidence indicator
    confidence_color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
    st.markdown(f"**Confidence:** {confidence_color} {confidence:.1%}")
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}** (Similarity: {source.score:.3f})")
            
            # Metadata
            meta = source.metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Product:** {meta.get('product', 'Unknown')}")
            with col2:
                st.markdown(f"**Category:** {meta.get('category', 'Unknown')}")
            with col3:
                st.markdown(f"**Issue:** {meta.get('issue', 'Unknown')}")
            
            # Source text
            st.markdown("**Text:**")
            st.markdown(f"> {source.text}")
            
            if i < len(sources):
                st.divider()


def display_session_stats():
    """Display session statistics in the sidebar."""
    stats = st.session_state.session_stats
    
    st.sidebar.markdown("### 📊 Session Statistics")
    st.sidebar.metric("Queries Processed", stats["queries_processed"])
    
    if stats["queries_processed"] > 0:
        avg_time = stats["total_processing_time"] / stats["queries_processed"]
        st.sidebar.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    session_duration = time.time() - stats["session_start"]
    st.sidebar.metric("Session Duration", f"{session_duration/60:.1f} min")


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="CrediTrust Complaint Analysis",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🏦 CrediTrust Complaint Analysis Assistant")
    st.markdown("""
    Welcome to the intelligent complaint analysis system. Ask questions about financial complaints 
    and get insights backed by real complaint data.
    """)
    
    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        example_queries = [
            "What are the most common credit card issues?",
            "Show me complaints about unauthorized charges",
            "What billing problems do customers face?",
            "Tell me about mortgage-related complaints",
            "What are common debt collection issues?"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            col = cols[i % 2]
            if col.button(f"📝 {query}", key=f"example_{i}"):
                st.session_state.example_query = query
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"]["answer"])
                if "sources" in message["content"]:
                    display_sources(
                        message["content"]["sources"], 
                        message["content"]["confidence"]
                    )
    
    # Handle example query selection
    if hasattr(st.session_state, 'example_query'):
        prompt = st.session_state.example_query
        del st.session_state.example_query
    else:
        prompt = None
    
    # Chat input
    if prompt or (prompt := st.chat_input("Ask about financial complaints...")):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing complaints..."):
                try:
                    response = st.session_state.rag_pipeline.run(prompt)
                    formatted_response = format_response_for_streamlit(response)
                    
                    # Display answer
                    st.markdown(formatted_response["answer"])
                    
                    # Display sources
                    display_sources(
                        formatted_response["sources"], 
                        formatted_response["confidence"]
                    )
                    
                    # Update session stats
                    stats = st.session_state.session_stats
                    stats["queries_processed"] += 1
                    stats["total_processing_time"] += response.processing_time
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_response
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": {"answer": error_msg, "sources": [], "confidence": 0.0}
                    })
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.session_stats = {
                "queries_processed": 0,
                "total_processing_time": 0.0,
                "session_start": time.time()
            }
            st.rerun()
        
        # Export chat button
        if st.button("📥 Export Chat", type="secondary"):
            if st.session_state.messages:
                chat_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": st.session_state.messages,
                    "stats": st.session_state.session_stats
                }
                
                st.download_button(
                    label="💾 Download Chat History",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"chat_history_{int(time.time())}.json",
                    mime="application/json"
                )
        
        st.divider()
        
        # Session statistics
        display_session_stats()
        
        st.divider()
        
        # Tips and information
        st.header("💡 Tips")
        st.markdown("""
        - **Be specific** in your questions
        - Ask about **products, issues, or trends**
        - Check the **sources** for verification
        - Use the **confidence score** as a guide
        - Try the **example queries** above
        """)
        
        st.header("🔍 Available Categories")
        try:
            if hasattr(st.session_state.rag_pipeline.retriever, 'vector_store'):
                categories = st.session_state.rag_pipeline.retriever.vector_store.get_available_categories()
                for cat in categories[:8]:  # Show first 8 categories
                    st.markdown(f"• {cat}")
                if len(categories) > 8:
                    st.markdown(f"• ... and {len(categories) - 8} more")
        except:
            st.markdown("Categories loading...")
        
        st.divider()
        
        # System information
        st.header("ℹ️ System Info")
        try:
            vector_store = st.session_state.rag_pipeline.retriever.vector_store
            stats = vector_store.get_stats()
            st.markdown(f"**Total Chunks:** {stats.get('total_chunks', 'Unknown')}")
            st.markdown(f"**Embedding Dim:** {stats.get('embedding_dim', 'Unknown')}")
        except:
            st.markdown("System info unavailable")


if __name__ == "__main__":
    main()