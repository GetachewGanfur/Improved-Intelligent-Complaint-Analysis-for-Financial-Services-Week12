import streamlit as st
import sys
import os

# --- Path setup for imports ---
# Ensure src is in sys.path for imports if running from project root
src_dir = os.path.abspath(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from rag_pipeline import create_simple_pipeline

def main():
    """Main Streamlit application function."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="CrediTrust Complaint Analysis Assistant", 
        page_icon="🏦", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏦 CrediTrust Complaint Analysis Assistant")
    st.markdown(
        """
        Welcome to the interactive chat assistant for financial complaint analysis!  
        Ask any question about financial complaints, trends, or issues, and our Retrieval-Augmented Generation (RAG) system will provide an answer based on real complaint data.
        """
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")
        use_mock = st.checkbox("Use Mock Generator (Fast, No LLM)", value=False)
        model_name = st.text_input("Model Name", value="deepseek-ai/DeepSeek-V3-0324")
        st.markdown("---")
        st.info("Using DeepSeek model for contextual answers. You can change the model or use a mock generator for testing.")

    # Initialize RAG pipeline in session state
    if "rag_pipeline" not in st.session_state:
        with st.spinner("Loading RAG pipeline..."):
            try:
                st.session_state["rag_pipeline"] = create_simple_pipeline()
                st.success("RAG system loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load RAG system: {e}")
                st.stop()

    rag_pipeline = st.session_state["rag_pipeline"]

    # Chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Chat input
    def submit_chat():
        user_input = st.session_state.get("user_input", "")
        if user_input.strip() == "":
            return
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("Generating answer..."):
            response = rag_pipeline.run(user_input)
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response.answer,
            "confidence": response.confidence_score,
            "sources": response.retrieved_sources,
            "processing_time": response.processing_time,
            "context_used": response.context_used
        })
        st.session_state["user_input"] = ""

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
            with st.expander("Show Details"):
                st.markdown(f"**Confidence Score:** {msg.get('confidence', 0):.2f}")
                st.markdown(f"**Processing Time:** {msg.get('processing_time', 0):.2f} seconds")
                st.markdown("**Sources Used:**")
                sources = msg.get("sources", [])
                if sources:
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"- **Source {i}**: *{src.metadata.get('product', 'Unknown')}* | "
                            f"Category: {src.metadata.get('category', 'Unknown')}, "
                            f"Issue: {src.metadata.get('issue', 'Unknown')}"
                        )
                else:
                    st.markdown("_No sources retrieved._")
                st.markdown("**Context Used:**")
                st.code(msg.get("context_used", ""), language="markdown")

    # User input box
    st.text_input(
        "Type your question and press Enter:",
        key="user_input",
        on_change=submit_chat,
        placeholder="E.g., What are common issues with credit card complaints?"
    )

    # Option to clear chat
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []

if __name__ == "__main__":
    main()
