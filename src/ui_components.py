"""
Reusable UI Components for Chat Interfaces

This module provides common UI components that can be used across
different interface implementations (Gradio, Streamlit, etc.).
"""

import time
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from rag_pipeline import RAGResponse, SearchResult


@dataclass
class UIConfig:
    """Configuration for UI components."""
    max_sources_display: int = 5
    max_text_preview: int = 200
    confidence_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.0
            }


class BaseUIFormatter(ABC):
    """Abstract base class for UI formatters."""
    
    def __init__(self, config: UIConfig = None):
        self.config = config or UIConfig()
    
    @abstractmethod
    def format_response(self, response: RAGResponse) -> Dict[str, Any]:
        """Format a RAG response for UI display."""
        pass
    
    @abstractmethod
    def format_sources(self, sources: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format source information for UI display."""
        pass
    
    def get_confidence_level(self, score: float) -> str:
        """Get confidence level based on score."""
        if score >= self.config.confidence_thresholds["high"]:
            return "high"
        elif score >= self.config.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def get_confidence_emoji(self, score: float) -> str:
        """Get emoji representation of confidence level."""
        level = self.get_confidence_level(score)
        return {
            "high": "🟢",
            "medium": "🟡", 
            "low": "🔴"
        }[level]
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to specified length."""
        max_length = max_length or self.config.max_text_preview
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class StandardUIFormatter(BaseUIFormatter):
    """Standard formatter for common UI patterns."""
    
    def format_response(self, response: RAGResponse) -> Dict[str, Any]:
        """Format a complete RAG response."""
        return {
            "answer": response.answer,
            "sources": self.format_sources(response.retrieved_sources),
            "confidence": {
                "score": response.confidence_score,
                "level": self.get_confidence_level(response.confidence_score),
                "emoji": self.get_confidence_emoji(response.confidence_score),
                "percentage": f"{response.confidence_score:.1%}"
            },
            "metadata": {
                "processing_time": response.processing_time,
                "query": response.query,
                "timestamp": time.time(),
                "sources_count": len(response.retrieved_sources)
            }
        }
    
    def format_sources(self, sources: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format source information with metadata."""
        formatted_sources = []
        
        # Limit number of sources displayed
        display_sources = sources[:self.config.max_sources_display]
        
        for i, source in enumerate(display_sources, 1):
            metadata = source.metadata or {}
            
            formatted_source = {
                "id": i,
                "text": source.text,
                "text_preview": self.truncate_text(source.text),
                "score": source.score,
                "score_formatted": f"{source.score:.3f}",
                "chunk_id": source.chunk_id,
                "metadata": {
                    "product": metadata.get("product", "Unknown"),
                    "category": metadata.get("category", "Unknown"),
                    "issue": metadata.get("issue", "Unknown"),
                    "company": metadata.get("company", "Unknown"),
                    "state": metadata.get("state", "Unknown"),
                    "date_received": metadata.get("date_received", "Unknown")
                }
            }
            
            formatted_sources.append(formatted_source)
        
        return formatted_sources


class ChatSessionManager:
    """Manages chat session state and history."""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.messages = []
        self.created_at = time.time()
        self.stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "total_sources_retrieved": 0
        }
    
    def add_message(self, role: str, content: Any, metadata: Dict = None):
        """Add a message to the session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Update stats for assistant messages
        if role == "assistant" and isinstance(content, dict):
            self._update_stats(content)
    
    def _update_stats(self, response_data: Dict):
        """Update session statistics."""
        self.stats["queries_processed"] += 1
        
        if "metadata" in response_data:
            metadata = response_data["metadata"]
            self.stats["total_processing_time"] += metadata.get("processing_time", 0)
            self.stats["total_sources_retrieved"] += metadata.get("sources_count", 0)
        
        if "confidence" in response_data:
            confidence = response_data["confidence"]["score"]
            current_avg = self.stats["average_confidence"]
            count = self.stats["queries_processed"]
            self.stats["average_confidence"] = (current_avg * (count - 1) + confidence) / count
    
    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """Get conversation history in tuple format."""
        history = []
        for i in range(0, len(self.messages), 2):
            if i + 1 < len(self.messages):
                user_msg = self.messages[i]["content"]
                assistant_msg = self.messages[i + 1]["content"]
                
                # Extract text content if it's a dict
                if isinstance(assistant_msg, dict):
                    assistant_msg = assistant_msg.get("answer", str(assistant_msg))
                
                history.append((user_msg, assistant_msg))
        
        return history
    
    def clear(self):
        """Clear the session."""
        self.messages.clear()
        self.stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "total_sources_retrieved": 0
        }
    
    def export_to_dict(self) -> Dict:
        """Export session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "stats": self.stats,
            "messages": self.messages
        }
    
    def export_to_json(self) -> str:
        """Export session to JSON string."""
        return json.dumps(self.export_to_dict(), indent=2, default=str)


class QuerySuggestionEngine:
    """Generates query suggestions based on available data."""
    
    def __init__(self):
        self.example_queries = [
            "What are the most common credit card issues?",
            "Show me complaints about unauthorized charges",
            "What billing problems do customers face?",
            "Tell me about mortgage-related complaints",
            "What are common debt collection issues?",
            "How do customers complain about bank account fees?",
            "What problems do people have with student loans?",
            "Show me issues with credit reporting",
            "What are the main payday loan complaints?",
            "Tell me about identity theft complaints"
        ]
        
        self.category_queries = {
            "Credit Card": [
                "What are common credit card billing issues?",
                "Show me credit card fraud complaints",
                "What problems do people have with credit card interest rates?"
            ],
            "Mortgage": [
                "What are the main mortgage servicing issues?",
                "Show me complaints about mortgage modifications",
                "What problems occur during the mortgage application process?"
            ],
            "Bank Account": [
                "What are common checking account problems?",
                "Show me complaints about overdraft fees",
                "What issues do people have with online banking?"
            ]
        }
    
    def get_random_suggestions(self, count: int = 5) -> List[str]:
        """Get random query suggestions."""
        import random
        return random.sample(self.example_queries, min(count, len(self.example_queries)))
    
    def get_category_suggestions(self, category: str) -> List[str]:
        """Get suggestions for a specific category."""
        return self.category_queries.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self.category_queries.keys())


class ResponseValidator:
    """Validates and scores response quality."""
    
    def __init__(self):
        self.min_answer_length = 10
        self.min_confidence_threshold = 0.1
        self.min_sources_count = 1
    
    def validate_response(self, response: RAGResponse) -> Dict[str, Any]:
        """Validate a RAG response and return quality metrics."""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "quality_score": 1.0,
            "recommendations": []
        }
        
        # Check answer length
        if len(response.answer) < self.min_answer_length:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Answer too short")
            validation_result["quality_score"] *= 0.5
        
        # Check confidence
        if response.confidence_score < self.min_confidence_threshold:
            validation_result["issues"].append("Low confidence score")
            validation_result["quality_score"] *= 0.7
            validation_result["recommendations"].append("Try rephrasing your question")
        
        # Check sources
        if len(response.retrieved_sources) < self.min_sources_count:
            validation_result["issues"].append("Insufficient sources")
            validation_result["quality_score"] *= 0.6
            validation_result["recommendations"].append("Try a more specific question")
        
        # Check for error indicators in answer
        error_indicators = ["error", "sorry", "couldn't", "unable", "failed"]
        if any(indicator in response.answer.lower() for indicator in error_indicators):
            validation_result["issues"].append("Potential error in response")
            validation_result["quality_score"] *= 0.3
        
        return validation_result


class UIThemeManager:
    """Manages UI themes and styling."""
    
    def __init__(self):
        self.themes = {
            "default": {
                "primary_color": "#1f77b4",
                "secondary_color": "#ff7f0e", 
                "success_color": "#2ca02c",
                "warning_color": "#ff7f0e",
                "error_color": "#d62728",
                "background_color": "#ffffff",
                "text_color": "#000000"
            },
            "dark": {
                "primary_color": "#4a90e2",
                "secondary_color": "#f5a623",
                "success_color": "#7ed321",
                "warning_color": "#f5a623",
                "error_color": "#d0021b",
                "background_color": "#1a1a1a",
                "text_color": "#ffffff"
            },
            "financial": {
                "primary_color": "#2c5aa0",
                "secondary_color": "#28a745",
                "success_color": "#28a745",
                "warning_color": "#ffc107",
                "error_color": "#dc3545",
                "background_color": "#f8f9fa",
                "text_color": "#212529"
            }
        }
        
        self.current_theme = "default"
    
    def get_theme(self, theme_name: str = None) -> Dict[str, str]:
        """Get theme configuration."""
        theme_name = theme_name or self.current_theme
        return self.themes.get(theme_name, self.themes["default"])
    
    def set_theme(self, theme_name: str):
        """Set the current theme."""
        if theme_name in self.themes:
            self.current_theme = theme_name
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes."""
        return list(self.themes.keys())


# Factory functions for easy component creation
def create_ui_formatter(config: UIConfig = None) -> StandardUIFormatter:
    """Create a standard UI formatter."""
    return StandardUIFormatter(config)


def create_session_manager(session_id: str = None) -> ChatSessionManager:
    """Create a chat session manager."""
    return ChatSessionManager(session_id)


def create_suggestion_engine() -> QuerySuggestionEngine:
    """Create a query suggestion engine."""
    return QuerySuggestionEngine()


def create_response_validator() -> ResponseValidator:
    """Create a response validator."""
    return ResponseValidator()


def create_theme_manager() -> UIThemeManager:
    """Create a theme manager."""
    return UIThemeManager()


if __name__ == "__main__":
    # Example usage
    print("🎨 UI Components Module")
    print("=" * 40)
    
    # Test formatter
    formatter = create_ui_formatter()
    print(f"✅ Formatter created with config: {formatter.config}")
    
    # Test session manager
    session = create_session_manager()
    print(f"✅ Session manager created: {session.session_id}")
    
    # Test suggestion engine
    suggestions = create_suggestion_engine()
    random_queries = suggestions.get_random_suggestions(3)
    print(f"✅ Sample suggestions: {random_queries}")
    
    # Test theme manager
    themes = create_theme_manager()
    available_themes = themes.get_available_themes()
    print(f"✅ Available themes: {available_themes}")
    
    print("\n🎯 UI Components ready for use!")