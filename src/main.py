"""
Main Application Module for Financial Complaint Analysis RAG System

This module provides a command-line interface for running the RAG pipeline
and evaluation system. It serves as the main entry point for the application.
"""

import os
import sys
import argparse
import logging
from typing import Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline, create_simple_pipeline
from rag_evaluator import RAGEvaluator, create_evaluator
from vector_store_utils import ComplaintVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialComplaintRAGApp:
    """Main application class for the Financial Complaint RAG system."""
    
    def __init__(self, vector_store_dir: str = "../vector_store"):
        self.vector_store_dir = vector_store_dir
        self.rag_pipeline = None
        self.evaluator = None
        
    def initialize(self) -> bool:
        """Initialize the RAG system components."""
        try:
            logger.info("Initializing Financial Complaint RAG System...")
            
            # Check if vector store exists
            if not os.path.exists(self.vector_store_dir):
                logger.error(f"Vector store directory not found: {self.vector_store_dir}")
                logger.error("Please run Task 2 notebook first to create the vector store.")
                return False
            
            # Initialize RAG pipeline
            logger.info("Loading RAG pipeline...")
            self.rag_pipeline = create_simple_pipeline(self.vector_store_dir)
            
            # Initialize evaluator
            logger.info("Loading evaluator...")
            self.evaluator = create_evaluator(self.vector_store_dir)
            
            logger.info("System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def run_interactive_query(self) -> None:
        """Run an interactive query session."""
        if not self.rag_pipeline:
            logger.error("RAG pipeline not initialized. Run initialize() first.")
            return
        
        print("\n" + "="*60)
        print("Financial Complaint Analysis RAG System")
        print("="*60)
        print("Type 'quit' to exit, 'help' for available commands")
        print("Type your question about financial complaints below:")
        print("-" * 60)
        
        while True:
            try:
                query = input("\nQuestion: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif query.lower() in ['help', 'h']:
                    self._show_help()
                    continue
                elif not query:
                    continue
                
                # Run RAG pipeline
                print("\n Processing your question...")
                response = self.rag_pipeline.run(query)
                
                # Display results
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"An error occurred: {e}")
    
    def _display_response(self, response) -> None:
        """Display the RAG response in a formatted way."""
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(response.answer)
        
        print(f"\nConfidence Score: {response.confidence_score:.3f}")
        print(f"Processing Time: {response.processing_time:.2f}s")
        print(f"Sources Retrieved: {len(response.retrieved_sources)}")
        
        if response.retrieved_sources:
            print("\n" + "-"*60)
            print("SOURCES")
            print("-"*60)
            
            for i, source in enumerate(response.retrieved_sources, 1):
                print(f"\nSource {i}:")
                print(f"  Category: {source.metadata.get('category', 'Unknown')}")
                print(f"  Product: {source.metadata.get('product', 'Unknown')}")
                print(f"  Similarity Score: {source.score:.3f}")
                print(f"  Text: {source.text[:200]}...")
        
        print("\n" + "="*60)
    
    def _show_help(self) -> None:
        """Show available commands and help information."""
        help_text = """
Available Commands:
- Type any question about financial complaints
- 'help' or 'h': Show this help message
- 'quit', 'exit', or 'q': Exit the application

Example Questions:
- What are the most common credit card billing issues?
- Which financial products have the most complaints?
- How do billing disputes typically get resolved?
- What are the main causes of unauthorized charges?
- Are there seasonal patterns in complaint volumes?

Tips:
- Be specific in your questions for better results
- The system will retrieve relevant complaint data and generate answers
- Check the source information to understand where answers come from
"""
        print(help_text)
    
    def run_evaluation(self, save_report: bool = True) -> None:
        """Run the full evaluation of the RAG pipeline."""
        if not self.evaluator:
            logger.error("Evaluator not initialized. Run initialize() first.")
            return
        
        try:
            logger.info("Starting RAG pipeline evaluation...")
            
            # Run evaluation on all questions
            results = self.evaluator.run_full_evaluation()
            
            # Display summary
            self._display_evaluation_summary(results)
            
            # Save report if requested
            if save_report:
                report_path = self.evaluator.save_evaluation_report(results)
                logger.info(f"Evaluation report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
    
    def _display_evaluation_summary(self, results) -> None:
        """Display a summary of evaluation results."""
        if not results:
            print("No evaluation results to display.")
            return
        
        total_questions = len(results)
        avg_quality = sum(r.quality_score for r in results) / total_questions
        avg_relevance = sum(r.relevance_score for r in results) / total_questions
        avg_completeness = sum(r.completeness_score for r in results) / total_questions
        avg_confidence = sum(r.confidence_score for r in results) / total_questions
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions Evaluated: {total_questions}")
        print(f"Average Quality Score: {avg_quality:.2f}/5.00")
        print(f"Average Relevance Score: {avg_relevance:.2f}/5.00")
        print(f"Average Completeness Score: {avg_completeness:.2f}/5.00")
        print(f"Average Confidence Score: {avg_confidence:.3f}")
        
        # Show top and bottom performers
        print(f"\nTop Performers (Quality Score = 5):")
        top_performers = [r for r in results if r.quality_score == 5]
        for result in top_performers[:3]:
            print(f"  - {result.question[:50]}...")
        
        print(f"\nAreas for Improvement (Quality Score ≤ 2):")
        low_performers = [r for r in results if r.quality_score <= 2]
        for result in low_performers[:3]:
            print(f"  - {result.question[:50]}...")
        
        print("="*60)
    
    def run_single_evaluation(self, question: str) -> None:
        """Run evaluation on a single custom question."""
        if not self.evaluator:
            logger.error("Evaluator not initialized. Run initialize() first.")
            return
        
        try:
            # Create a custom evaluation question
            from rag_evaluator import EvaluationQuestion
            
            eval_question = EvaluationQuestion(
                question=question,
                expected_topics=[],  # Empty for custom questions
                expected_sources=3,
                difficulty="custom",
                category="custom"
            )
            
            # Run evaluation
            result = self.evaluator.evaluate_single_question(eval_question)
            
            # Display result
            print(f"\nQuestion: {result.question}")
            print(f"Answer: {result.generated_answer}")
            print(f"Quality Score: {result.quality_score}/5")
            print(f"Relevance Score: {result.relevance_score}/5")
            print(f"Completeness Score: {result.completeness_score}/5")
            print(f"Comments: {result.comments}")
            
        except Exception as e:
            logger.error(f"Error during single evaluation: {e}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Financial Complaint Analysis RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive query mode
  python main.py --interactive
  
  # Run full evaluation
  python main.py --evaluate
  
  # Run evaluation on custom question
  python main.py --evaluate-question "What are credit card fraud patterns?"
  
  # Run both evaluation and interactive mode
  python main.py --evaluate --interactive
        """
    )
    
    parser.add_argument(
        '--vector-store-dir',
        default='../vector_store',
        help='Path to vector store directory (default: ../vector_store)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run interactive query mode'
    )
    
    parser.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='Run full RAG pipeline evaluation'
    )
    
    parser.add_argument(
        '--evaluate-question',
        metavar='QUESTION',
        help='Run evaluation on a specific question'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip saving evaluation report when running evaluation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and initialize application
    app = FinancialComplaintRAGApp(args.vector_store_dir)
    
    if not app.initialize():
        sys.exit(1)
    
    # Run requested operations
    if args.evaluate:
        logger.info("Running RAG pipeline evaluation...")
        app.run_evaluation(save_report=not args.no_report)
    
    if args.evaluate_question:
        logger.info(f"Evaluating custom question: {args.evaluate_question}")
        app.run_single_evaluation(args.evaluate_question)
    
    if args.interactive:
        logger.info("Starting interactive query mode...")
        app.run_interactive_query()
    
    # If no specific mode specified, run interactive by default
    if not any([args.evaluate, args.evaluate_question, args.interactive]):
        logger.info("No specific mode specified, starting interactive query mode...")
        app.run_interactive_query()


if __name__ == "__main__":
    main()