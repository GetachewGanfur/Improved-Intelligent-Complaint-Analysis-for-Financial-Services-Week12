"""
Interactive Chat Interface for Financial Complaint Analysis

This module provides a modular and object-oriented chat interface using Gradio
for interacting with the RAG system. It includes:
- ChatSession: Manages conversation state and history
- ResponseFormatter: Formats responses with sources
- ChatInterface: Main interface controller
- StreamingHandler: Handles response streaming (optional) 
"""

import time
import json
import logging
from typing import List, Dict, Optional, Tuple, Generator, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import gradio as gr
import pandas as pd

from .rag_pipeline import RAGPipeline, RAGResponse, create_simple_pipeline
from .vector_store_utils import ComplaintVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    sources: Optional[List[Dict]] = None
    confidence: Optional[float] = None


@dataclass
class ChatSession:
    """Manages chat session state and history."""
    session_id: str
    messages: List[ChatMessage]
    created_at: float
    
    def add_message(self, role: str, content: str, sources: Optional[List[Dict]] = None, 
                   confidence: Optional[float] = None) -> None:
        """Add a message to the session."""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            sources=sources,
            confidence=confidence
        )
        self.messages.append(message)
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Get chat history in Gradio format."""
        history = []
        for i in range(0, len(self.messages), 2):
            if i + 1 < len(self.messages):
                user_msg = self.messages[i].content
                assistant_msg = self.messages[i + 1].content
                history.append((user_msg, assistant_msg))
        return history
    
    def clear(self) -> None:
        """Clear the chat session."""
        self.messages.clear()
    
    def export_to_dict(self) -> Dict:
        """Export session to dictionary for saving."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'messages': [asdict(msg) for msg in self.messages]
        }


class BaseResponseFormatter(ABC):
    """Abstract base class for response formatting."""
    
    @abstractmethod
    def format_response(self, response: RAGResponse) -> str:
        """Format the RAG response for display."""
        pass
    
    @abstractmethod
    def format_sources(self, sources: List[Dict]) -> str:
        """Format source information for display."""
        pass


class GradioResponseFormatter(BaseResponseFormatter):
    """Response formatter optimized for Gradio interface."""
    
    def format_response(self, response: RAGResponse) -> str:
        """Format the complete response with answer and sources."""
        formatted_parts = []
        
        # Main answer
        formatted_parts.append(f"**Answer:**\n{response.answer}")
        
        # Confidence indicator
        if response.confidence_score > 0:
            confidence_emoji = self._get_confidence_emoji(response.confidence_score)
            formatted_parts.append(f"\n**Confidence:** {confidence_emoji} {response.confidence_score:.1%}")
        
        # Sources
        if response.retrieved_sources:
            sources_text = self.format_sources(response.retrieved_sources)
            formatted_parts.append(f"\n**Sources:**\n{sources_text}")
        
        # Processing time
        formatted_parts.append(f"\n*Response generated in {response.processing_time:.2f}s*")
        
        return "\n".join(formatted_parts)
    
    def format_sources(self, sources: List[Dict]) -> str:
        """Format source information with metadata."""
        if not sources:
            return "No sources available."
        
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            metadata = source.get('metadata', {})
            
            # Create source header
            source_header = f"**Source {i}** (Similarity: {source.get('score', 0):.3f})"
            
            # Add metadata
            meta_info = []
            if metadata.get('product'):
                meta_info.append(f"Product: {metadata['product']}")
            if metadata.get('category'):
                meta_info.append(f"Category: {metadata['category']}")
            if metadata.get('issue'):
                meta_info.append(f"Issue: {metadata['issue']}")
            
            meta_text = " | ".join(meta_info) if meta_info else "No metadata available"
            
            # Format text (truncate if too long)
            text = source.get('text', '')
            if len(text) > 300:
                text = text[:300] + "..."
            
            formatted_source = f"{source_header}\n*{meta_text}*\n> {text}\n"
            formatted_sources.append(formatted_source)
        
        return "\n".join(formatted_sources)
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji based on confidence level."""
        if confidence >= 0.8:
            return "🟢"
        elif confidence >= 0.6:
            return "🟡"
        else:
            return "🔴"


class StreamingHandler:
    """Handles response streaming for better user experience."""
    
    def __init__(self, formatter: BaseResponseFormatter):
        self.formatter = formatter
    
    def stream_response(self, response: RAGResponse) -> Generator[str, None, None]:
        """Stream the response token by token."""
        full_response = self.formatter.format_response(response)
        
        # Simulate streaming by yielding parts of the response
        words = full_response.split()
        current_text = ""
        
        for word in words:
            current_text += word + " "
            yield current_text
            time.sleep(0.05)  # Small delay for streaming effect
        
        yield full_response  # Final complete response


class ChatInterface:
    """Main chat interface controller."""
    
    def __init__(self, 
                 rag_pipeline: RAGPipeline,
                 formatter: BaseResponseFormatter = None,
                 enable_streaming: bool = False):
        self.rag_pipeline = rag_pipeline
        self.formatter = formatter or GradioResponseFormatter()
        self.enable_streaming = enable_streaming
        self.streaming_handler = StreamingHandler(self.formatter) if enable_streaming else None
        
        # Session management
        self.current_session = ChatSession(
            session_id=f"session_{int(time.time())}",
            messages=[],
            created_at=time.time()
        )
        
        logger.info("Chat interface initialized")
    
    def process_query(self, query: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Process user query and return response with updated history."""
        if not query.strip():
            return "", history
        
        try:
            # Add user message to session
            self.current_session.add_message("user", query)
            
            # Get RAG response
            logger.info(f"Processing query: {query[:50]}...")
            response = self.rag_pipeline.run(query)
            
            # Format response
            formatted_response = self.formatter.format_response(response)
            
            # Add assistant message to session
            sources_dict = [
                {
                    'text': src.text,
                    'score': src.score,
                    'metadata': src.metadata,
                    'chunk_id': src.chunk_id
                }
                for src in response.retrieved_sources
            ]
            
            self.current_session.add_message(
                "assistant", 
                formatted_response,
                sources=sources_dict,
                confidence=response.confidence_score
            )
            
            # Update history
            updated_history = history + [(query, formatted_response)]
            
            logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
            return "", updated_history
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            logger.error(f"Error processing query: {e}")
            
            # Add error to session
            self.current_session.add_message("user", query)
            self.current_session.add_message("assistant", error_msg)
            
            updated_history = history + [(query, error_msg)]
            return "", updated_history
    
    def clear_chat(self) -> Tuple[List[Tuple[str, str]], str]:
        """Clear the chat history."""
        self.current_session.clear()
        logger.info("Chat history cleared")
        return [], ""
    
    def get_session_stats(self) -> str:
        """Get current session statistics."""
        stats = {
            "Session ID": self.current_session.session_id,
            "Messages": len(self.current_session.messages),
            "Created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.current_session.created_at))
        }
        
        return "\n".join([f"**{k}:** {v}" for k, v in stats.items()])
    
    def export_session(self) -> str:
        """Export current session to JSON string."""
        try:
            session_data = self.current_session.export_to_dict()
            return json.dumps(session_data, indent=2)
        except Exception as e:
            return f"Error exporting session: {str(e)}"


class GradioApp:
    """Main Gradio application wrapper."""
    
    def __init__(self, 
                 vector_store_dir: str = "../vector_store",
                 enable_streaming: bool = False,
                 share: bool = False):
        self.vector_store_dir = vector_store_dir
        self.enable_streaming = enable_streaming
        self.share = share
        
        # Initialize components
        self.rag_pipeline = None
        self.chat_interface = None
        self.app = None
        
    def _initialize_pipeline(self):
        """Initialize the RAG pipeline."""
        try:
            logger.info("Initializing RAG pipeline...")
            self.rag_pipeline = create_simple_pipeline(self.vector_store_dir)
            self.chat_interface = ChatInterface(
                self.rag_pipeline,
                enable_streaming=self.enable_streaming
            )
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def _create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="CrediTrust Complaint Analysis Assistant",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-message {
                padding: 10px;
                margin: 5px 0;
                border-radius: 10px;
            }
            """
        ) as app:
            
            # Header
            gr.Markdown("""
            # 🏦 CrediTrust Complaint Analysis Assistant
            
            Welcome to the intelligent complaint analysis system. Ask questions about financial complaints 
            and get insights backed by real complaint data.
            
            **Example queries:**
            - "What are the most common credit card issues?"
            - "Show me complaints about unauthorized charges"
            - "What billing problems do customers face?"
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder="Ask a question about financial complaints...",
                            label="Your Question",
                            lines=2,
                            max_lines=5,
                            scale=4
                        )
                        submit_btn = gr.Button("Ask", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                        export_btn = gr.Button("Export Session", variant="secondary")
                
                with gr.Column(scale=1):
                    # Sidebar with information
                    gr.Markdown("### 📊 Session Info")
                    session_info = gr.Markdown(self.chat_interface.get_session_stats() if self.chat_interface else "Not initialized")
                    
                    gr.Markdown("### 💡 Tips")
                    gr.Markdown("""
                    - Be specific in your questions
                    - Ask about products, issues, or trends
                    - Check the sources for verification
                    - Use the confidence score as a guide
                    """)
                    
                    gr.Markdown("### 🔍 Available Categories")
                    if self.chat_interface and hasattr(self.chat_interface.rag_pipeline.retriever, 'vector_store'):
                        try:
                            categories = self.chat_interface.rag_pipeline.retriever.vector_store.get_available_categories()
                            categories_text = "\\n".join([f"- {cat}" for cat in categories[:10]])
                            gr.Markdown(categories_text)
                        except:
                            gr.Markdown("Categories not available")
                    else:
                        gr.Markdown("Categories loading...")
            
            # Export modal
            with gr.Row(visible=False) as export_row:
                export_output = gr.Textbox(
                    label="Session Export (JSON)",
                    lines=10,
                    max_lines=20
                )
            
            # Event handlers
            def submit_query(query, history):
                if self.chat_interface:
                    return self.chat_interface.process_query(query, history)
                return "", history
            
            def clear_chat():
                if self.chat_interface:
                    return self.chat_interface.clear_chat()
                return [], ""
            
            def export_session():
                if self.chat_interface:
                    return self.chat_interface.export_session()
                return "No session to export"
            
            def update_session_info():
                if self.chat_interface:
                    return self.chat_interface.get_session_stats()
                return "Not initialized"
            
            # Wire up events
            submit_btn.click(
                fn=submit_query,
                inputs=[query_input, chatbot],
                outputs=[query_input, chatbot]
            ).then(
                fn=update_session_info,
                outputs=[session_info]
            )
            
            query_input.submit(
                fn=submit_query,
                inputs=[query_input, chatbot],
                outputs=[query_input, chatbot]
            ).then(
                fn=update_session_info,
                outputs=[session_info]
            )
            
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, query_input]
            ).then(
                fn=update_session_info,
                outputs=[session_info]
            )
            
            export_btn.click(
                fn=export_session,
                outputs=[export_output]
            )
        
        return app
    
    def launch(self, **kwargs):
        """Launch the Gradio application."""
        # Initialize pipeline
        self._initialize_pipeline()
        
        # Create interface
        self.app = self._create_interface()
        
        # Launch with default settings
        launch_kwargs = {
            'share': self.share,
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'show_error': True,
            **kwargs
        }
        
        logger.info(f"Launching Gradio app with settings: {launch_kwargs}")
        return self.app.launch(**launch_kwargs)


def create_chat_app(vector_store_dir: str = "../vector_store", 
                   enable_streaming: bool = False,
                   share: bool = False) -> GradioApp:
    """Factory function to create a chat application."""
    return GradioApp(
        vector_store_dir=vector_store_dir,
        enable_streaming=enable_streaming,
        share=share
    )


if __name__ == "__main__":
    # Example usage
    print("Starting CrediTrust Complaint Analysis Assistant...")
    
    try:
        app = create_chat_app(
            vector_store_dir="../vector_store",
            enable_streaming=False,
            share=False
        )
        
        app.launch()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        logger.error(f"Application startup failed: {e}")