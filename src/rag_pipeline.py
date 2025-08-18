"""
RAG Pipeline for Financial Complaint Analysis

This module implements the core RAG (Retrieval-Augmented Generation) pipeline
for analyzing financial complaints. It includes:
- Retriever: Semantic search against the vector store
- Prompt Engineering: Robust templates for LLM guidance
- Generator: LLM integration for answer generation
- Pipeline Orchestration: End-to-end RAG workflow 
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
from sentence_transformers import SentenceTransformer

# Import our vector store utilities
from vector_store_utils import ComplaintVectorStore


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    top_k: int = 5
    similarity_threshold: float = 0.3
    max_context_length: int = 2000
    temperature: float = 0.7
    max_tokens: int = 500
    model_name: str = "microsoft/DialoGPT-medium"  # Default model


@dataclass
class SearchResult:
    """Represents a single search result from the vector store."""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: int


@dataclass
class RAGResponse:
    """Represents the complete RAG pipeline response."""
    answer: str
    retrieved_sources: List[SearchResult]
    confidence_score: float
    processing_time: float
    query: str
    context_used: str


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[SearchResult]:
        """Retrieve relevant documents for a query."""
        pass


class VectorStoreRetriever(BaseRetriever):
    """Retriever implementation using the complaint vector store."""
    
    def __init__(self, vector_store: ComplaintVectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int) -> List[SearchResult]:
        """Retrieve top-k most relevant chunks for the query."""
        try:
            results = self.vector_store.search(query, k=k)
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    text=result['text'],
                    score=result['score'],
                    metadata=result['metadata'],
                    chunk_id=result['chunk_id']
                )
                search_results.append(search_result)
            
            logger.info(f"Retrieved {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []


class BasePromptEngine(ABC):
    """Abstract base class for prompt engineering."""
    
    @abstractmethod
    def create_prompt(self, query: str, context: str) -> str:
        """Create a prompt from query and context."""
        pass


class FinancialComplaintPromptEngine(BasePromptEngine):
    """Specialized prompt engine for financial complaint analysis."""
    
    def __init__(self):
        self.base_template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based on the provided context.

IMPORTANT GUIDELINES:
1. Use ONLY the information provided in the context below
2. If the context doesn't contain enough information to answer the question, clearly state "I don't have enough information to answer this question based on the provided context."
3. Be specific and cite relevant details from the context
4. Maintain a professional, helpful tone
5. Focus on actionable insights when possible

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    def create_prompt(self, query: str, context: str) -> str:
        """Create a prompt from query and context."""
        return self.base_template.format(
            context=context,
            question=query
        )
    
    def create_analysis_prompt(self, query: str, context: str) -> str:
        """Create a more detailed analysis prompt."""
        analysis_template = """You are a senior financial analyst at CrediTrust. Analyze the following customer complaint data and provide comprehensive insights.

ANALYSIS REQUEST:
{question}

COMPLAINT DATA:
{context}

Please provide:
1. Key issues identified
2. Potential root causes
3. Recommendations for resolution
4. Risk assessment (if applicable)

ANALYSIS:"""
        
        return analysis_template.format(
            context=context,
            question=query
        )


class BaseGenerator(ABC):
    """Abstract base class for text generation."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text based on the prompt."""
        pass


class HuggingFaceGenerator(BaseGenerator):
    """Text generator using Hugging Face transformers."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        try:
            from transformers import pipeline
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=-1  # Use CPU by default
            )
            self.model_name = model_name
            logger.info(f"Loaded Hugging Face model: {model_name}")
        except ImportError:
            logger.error("transformers library not available. Please install it.")
            raise
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """Generate text based on the prompt."""
        try:
            # Generate response
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the original prompt
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I apologize, but I couldn't generate a meaningful response."
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return f"Error generating response: {str(e)}"


class MockGenerator(BaseGenerator):
    """Mock generator for testing and development without requiring heavy models."""
    
    def __init__(self):
        self.responses = {
            "credit card": "Based on the complaint data, I can see several credit card billing issues including unauthorized charges and incorrect interest calculations. The most common problems appear to be related to billing disputes and fee assessments.",
            "billing": "The complaint data shows multiple billing-related issues across different financial products. Common problems include incorrect charges, late fee assessments, and billing statement errors.",
            "unauthorized": "Several complaints mention unauthorized transactions and charges. This appears to be a significant concern that requires immediate attention from financial institutions.",
            "default": "Based on the provided context, I can analyze the customer complaint data to help answer your question. The information shows various financial service issues that customers have experienced."
        }
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response based on prompt content."""
        prompt_lower = prompt.lower()
        
        for key, response in self.responses.items():
            if key in prompt_lower:
                return response
        
        return self.responses["default"]


class RAGPipeline:
    """Main RAG pipeline orchestrating retrieval and generation."""
    
    def __init__(self, 
                 retriever: BaseRetriever,
                 prompt_engine: BasePromptEngine,
                 generator: BaseGenerator,
                 config: RAGConfig = None):
        self.retriever = retriever
        self.prompt_engine = prompt_engine
        self.generator = generator
        self.config = config or RAGConfig()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context from search results."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            # Add metadata information
            meta = result.metadata
            context_parts.append(f"Source {i}:")
            context_parts.append(f"Product: {meta.get('product', 'Unknown')}")
            context_parts.append(f"Category: {meta.get('category', 'Unknown')}")
            context_parts.append(f"Issue: {meta.get('issue', 'Unknown')}")
            context_parts.append(f"Text: {result.text}")
            context_parts.append("")  # Empty line for separation
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length] + "... [truncated]"
        
        return context
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence score based on search results."""
        if not search_results:
            return 0.0
        
        # Average similarity scores
        avg_score = sum(result.score for result in search_results) / len(search_results)
        
        # Boost confidence if we have multiple high-quality results
        if len(search_results) >= 3:
            avg_score *= 1.1
        
        return min(avg_score, 1.0)
    
    def run(self, query: str, k: Optional[int] = None) -> RAGResponse:
        """Run the complete RAG pipeline."""
        import time
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            k = k or self.config.top_k
            search_results = self.retriever.retrieve(query, k)
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    retrieved_sources=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    query=query,
                    context_used="No context available"
                )
            
            # Step 2: Prepare context
            context = self._prepare_context(search_results)
            
            # Step 3: Create prompt
            prompt = self.prompt_engine.create_prompt(query, context)
            
            # Step 4: Generate answer
            answer = self.generator.generate(prompt)
            
            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(search_results)
            
            processing_time = time.time() - start_time
            
            logger.info(f"RAG pipeline completed in {processing_time:.2f}s")
            
            return RAGResponse(
                answer=answer,
                retrieved_sources=search_results,
                confidence_score=confidence,
                processing_time=processing_time,
                query=query,
                context_used=context
            )
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return RAGResponse(
                answer=f"An error occurred while processing your request: {str(e)}",
                retrieved_sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                query=query,
                context_used="Error occurred"
            )


class RAGPipelineFactory:
    """Factory class for creating RAG pipeline instances."""
    
    @staticmethod
    def create_pipeline(vector_store_dir: str = None,
                       model_name: str = "microsoft/DialoGPT-medium",
                       use_mock_generator: bool = False) -> RAGPipeline:
        if vector_store_dir is None:
            # Get the correct path to vector store
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vector_store_dir = os.path.join(os.path.dirname(current_dir), "vector_store")
        """Create a complete RAG pipeline instance."""
        
        # Load vector store
        vector_store = ComplaintVectorStore(vector_store_dir)
        vector_store.load()
        
        # Create components
        retriever = VectorStoreRetriever(vector_store)
        prompt_engine = FinancialComplaintPromptEngine()
        
        if use_mock_generator:
            generator = MockGenerator()
        else:
            generator = HuggingFaceGenerator(model_name)
        
        # Create and return pipeline
        return RAGPipeline(retriever, prompt_engine, generator)


def create_simple_pipeline(vector_store_dir: str = None) -> RAGPipeline:
    """Convenience function to create a simple RAG pipeline."""
    return RAGPipelineFactory.create_pipeline(
        vector_store_dir=vector_store_dir,
        use_mock_generator=True  # Use mock for simplicity
    )


if __name__ == "__main__":
    # Example usage
    print("Testing RAG Pipeline...")
    
    try:
        # Create pipeline
        pipeline = create_simple_pipeline()
        
        # Test query
        test_query = "What are the most common credit card billing issues?"
        response = pipeline.run(test_query)
        
        print(f"\nQuery: {test_query}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Processing time: {response.processing_time:.2f}s")
        print(f"Sources retrieved: {len(response.retrieved_sources)}")
        
        print("\nRAG Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Error testing RAG Pipeline: {e}")