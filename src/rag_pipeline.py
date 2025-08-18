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


class SimpleContextGenerator(BaseGenerator):
    """Enhanced context-based generator that provides comprehensive analysis."""
    
    def generate(self, prompt: str) -> str:
        """Generate response by analyzing the context in the prompt."""
        try:
            # Extract context and question from prompt
            if "CONTEXT:" in prompt and "QUESTION:" in prompt:
                context_start = prompt.find("CONTEXT:") + len("CONTEXT:")
                question_start = prompt.find("QUESTION:") + len("QUESTION:")
                answer_start = prompt.find("ANSWER:")
                
                context = prompt[context_start:question_start-len("QUESTION:")].strip()
                question = prompt[question_start:answer_start-len("ANSWER:")].strip()
                
                return self._analyze_context(context, question)
            else:
                return "I need properly formatted context to provide a meaningful answer."
                
        except Exception as e:
            return f"I encountered an issue processing your question: {str(e)}"
    
    def _analyze_context(self, context: str, question: str) -> str:
        """Analyze context to generate comprehensive response."""
        if not context or "No context available" in context:
            return "I don't have enough information in the complaint database to answer your question. Please try rephrasing your question or asking about specific financial products or issues."
        
        # Extract key information from context
        sources = context.split("Source ")
        if len(sources) < 2:
            return "I couldn't find relevant complaint information to answer your question. Please try asking about specific financial products like credit cards, mortgages, or checking accounts."
        
        # Parse structured data from sources
        parsed_data = self._parse_sources(sources[1:])
        
        if not parsed_data:
            return "I found some complaint data but couldn't extract meaningful information. Please try a more specific question."
        
        # Generate comprehensive response based on question type
        return self._generate_comprehensive_response(question, parsed_data)
    
    def _parse_sources(self, sources: List[str]) -> Dict:
        """Parse source data into structured format."""
        products = []
        categories = []
        issues = []
        complaint_texts = []
        subcategories = []
        
        for source in sources:
            lines = source.strip().split('\n')
            source_data = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Product:'):
                    product = line.replace('Product:', '').strip()
                    products.append(product)
                    source_data['product'] = product
                elif line.startswith('Category:'):
                    category = line.replace('Category:', '').strip()
                    categories.append(category)
                    source_data['category'] = category
                elif line.startswith('Issue:'):
                    issue = line.replace('Issue:', '').strip()
                    issues.append(issue)
                    source_data['issue'] = issue
                elif line.startswith('Text:'):
                    text = line.replace('Text:', '').strip()
                    complaint_texts.append(text)
                    source_data['text'] = text
            
            if source_data.get('text'):
                # Extract subcategories from text analysis
                subcats = self._extract_subcategories(source_data['text'])
                subcategories.extend(subcats)
        
        return {
            'products': products,
            'categories': categories,
            'issues': issues,
            'texts': complaint_texts,
            'subcategories': subcategories,
            'source_count': len([s for s in sources if s.strip()])
        }
    
    def _extract_subcategories(self, text: str) -> List[str]:
        """Extract specific issue subcategories from complaint text."""
        text_lower = text.lower()
        subcategories = []
        
        # Define subcategory patterns
        patterns = {
            'unauthorized_transactions': ['unauthorized', 'fraud', 'stolen', 'identity theft', 'not authorized'],
            'billing_errors': ['wrong amount', 'incorrect charge', 'billing error', 'overcharged', 'double charged'],
            'payment_processing': ['payment not processed', 'payment failed', 'autopay', 'payment declined'],
            'customer_service': ['rude', 'unhelpful', 'long wait', 'poor service', 'representative'],
            'account_closure': ['closed account', 'account closure', 'terminate account'],
            'fee_disputes': ['unexpected fee', 'hidden fee', 'excessive fee', 'fee waiver'],
            'credit_reporting': ['credit report', 'credit score', 'credit bureau', 'dispute'],
            'loan_modification': ['loan modification', 'refinance', 'payment plan', 'hardship'],
            'debt_collection': ['collection', 'debt collector', 'harassment', 'validation']
        }
        
        for category, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                subcategories.append(category.replace('_', ' ').title())
        
        return subcategories[:2]  # Return top 2 subcategories
    
    def _generate_comprehensive_response(self, question: str, data: Dict) -> str:
        """Generate a comprehensive response based on question type and data."""
        question_lower = question.lower()
        response_parts = []
        
        # Determine question type and generate appropriate response
        if any(word in question_lower for word in ['common', 'most', 'frequent', 'typical', 'main']):
            response_parts.extend(self._handle_frequency_question(question_lower, data))
        
        elif any(word in question_lower for word in ['how', 'what', 'why', 'when', 'where']):
            response_parts.extend(self._handle_descriptive_question(question_lower, data))
        
        elif any(word in question_lower for word in ['resolve', 'fix', 'solution', 'handle']):
            response_parts.extend(self._handle_resolution_question(question_lower, data))
        
        else:
            # General analysis
            response_parts.extend(self._handle_general_question(question_lower, data))
        
        # Add specific product/category insights
        response_parts.extend(self._add_specific_insights(question_lower, data))
        
        # Add data summary
        response_parts.append(f"\n\nThis analysis is based on {data['source_count']} relevant complaint records from our database.")
        
        if not response_parts:
            return "I found relevant complaint data but couldn't generate a specific answer. Please try asking a more focused question about financial products or specific issues."
        
        return " ".join(response_parts)
    
    def _handle_frequency_question(self, question: str, data: Dict) -> List[str]:
        """Handle questions about frequency/commonality."""
        parts = []
        
        if data['products']:
            product_counts = {}
            for product in data['products']:
                product_counts[product] = product_counts.get(product, 0) + 1
            
            top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_products:
                products_str = ", ".join([f"{prod} ({count} complaints)" for prod, count in top_products])
                parts.append(f"Based on the complaint data, the most frequently mentioned products are: {products_str}.")
        
        if data['categories']:
            category_counts = {}
            for category in data['categories']:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_categories:
                categories_str = ", ".join([f"{cat} ({count} cases)" for cat, count in top_categories])
                parts.append(f"The most common complaint categories are: {categories_str}.")
        
        if data['subcategories']:
            unique_subcats = list(set(data['subcategories']))
            if unique_subcats:
                parts.append(f"Key issue types identified include: {', '.join(unique_subcats[:4])}.")
        
        return parts
    
    def _handle_descriptive_question(self, question: str, data: Dict) -> List[str]:
        """Handle descriptive questions (what, how, why, etc.)."""
        parts = []
        
        # Analyze complaint texts for patterns
        if data['texts']:
            # Look for specific patterns based on question keywords
            if 'billing' in question or 'charge' in question:
                billing_issues = [text for text in data['texts'] if any(word in text.lower() for word in ['bill', 'charge', 'fee', 'payment'])]
                if billing_issues:
                    parts.append("Regarding billing issues, customers commonly report problems with incorrect charges, unexpected fees, and payment processing errors.")
            
            if 'credit card' in question:
                cc_issues = [text for text in data['texts'] if 'credit' in text.lower()]
                if cc_issues:
                    parts.append("Credit card complaints typically involve unauthorized transactions, billing disputes, interest rate concerns, and customer service issues.")
            
            if 'mortgage' in question or 'loan' in question:
                loan_issues = [text for text in data['texts'] if any(word in text.lower() for word in ['mortgage', 'loan', 'payment'])]
                if loan_issues:
                    parts.append("Mortgage and loan complaints often center around payment processing, modification requests, and communication issues with servicers.")
        
        # Add general insights
        if data['issues']:
            unique_issues = list(set(data['issues']))
            if unique_issues:
                parts.append(f"The main issues reported include: {', '.join(unique_issues[:4])}.")
        
        return parts
    
    def _handle_resolution_question(self, question: str, data: Dict) -> List[str]:
        """Handle questions about resolution/solutions."""
        parts = []
        
        parts.append("Based on the complaint patterns, here are common resolution approaches:")
        
        if any('billing' in text.lower() for text in data['texts']):
            parts.append("• For billing issues: Review account statements, contact customer service for charge disputes, and request fee waivers when appropriate.")
        
        if any('unauthorized' in text.lower() for text in data['texts']):
            parts.append("• For unauthorized transactions: Immediately report to the financial institution, file fraud claims, and monitor credit reports.")
        
        if any('service' in text.lower() for text in data['texts']):
            parts.append("• For customer service issues: Escalate to supervisors, document all interactions, and consider filing complaints with regulatory agencies.")
        
        parts.append("• General advice: Keep detailed records, know your rights, and don't hesitate to escalate unresolved issues to regulatory bodies like the CFPB.")
        
        return parts
    
    def _handle_general_question(self, question: str, data: Dict) -> List[str]:
        """Handle general questions."""
        parts = []
        
        if data['products'] and data['categories']:
            parts.append(f"The complaint data shows issues across {len(set(data['products']))} different financial products, with {len(set(data['categories']))} main complaint categories.")
        
        if data['subcategories']:
            unique_subcats = list(set(data['subcategories']))
            parts.append(f"Common issue types include: {', '.join(unique_subcats[:3])}.")
        
        return parts
    
    def _add_specific_insights(self, question: str, data: Dict) -> List[str]:
        """Add specific insights based on question keywords."""
        parts = []
        
        # Security-related insights
        if any(word in question for word in ['security', 'fraud', 'unauthorized', 'stolen']):
            security_texts = [text for text in data['texts'] if any(word in text.lower() for word in ['fraud', 'unauthorized', 'stolen', 'identity'])]
            if security_texts:
                parts.append("Security concerns are prevalent, with customers reporting unauthorized transactions and identity theft attempts.")
        
        # Fee-related insights
        if any(word in question for word in ['fee', 'charge', 'cost']):
            fee_texts = [text for text in data['texts'] if any(word in text.lower() for word in ['fee', 'charge', 'cost'])]
            if fee_texts:
                parts.append("Fee-related complaints often involve unexpected charges, lack of transparency in fee structures, and difficulty obtaining fee waivers.")
        
        return parts


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
            generator = SimpleContextGenerator()
        else:
            generator = HuggingFaceGenerator(model_name)
        
        # Create and return pipeline
        return RAGPipeline(retriever, prompt_engine, generator)


def create_simple_pipeline(vector_store_dir: str = None) -> RAGPipeline:
    """Convenience function to create a simple RAG pipeline."""
    if vector_store_dir is None:
        # Get the correct path to vector store
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vector_store_dir = os.path.join(os.path.dirname(current_dir), "vector_store")
    
    # Load vector store
    vector_store = ComplaintVectorStore(vector_store_dir)
    vector_store.load()
    
    # Create components
    retriever = VectorStoreRetriever(vector_store)
    prompt_engine = FinancialComplaintPromptEngine()
    generator = SimpleContextGenerator()  # Use context-aware generator
    
    # Create and return pipeline
    return RAGPipeline(retriever, prompt_engine, generator)


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