"""
RAG Pipeline Evaluation Module

This module provides comprehensive evaluation capabilities for the RAG pipeline,
including qualitative assessment, scoring, and detailed analysis of results.
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from .rag_pipeline import RAGPipeline, RAGResponse, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuestion:
    """Represents a test question for RAG evaluation."""
    question: str
    expected_topics: List[str]  # Key topics that should be mentioned
    expected_sources: int  # Expected number of relevant sources
    difficulty: str  # 'easy', 'medium', 'hard'
    category: str  # Question category for analysis


@dataclass
class EvaluationResult:
    """Represents the evaluation result for a single question."""
    question: str
    generated_answer: str
    retrieved_sources: List[Dict]
    confidence_score: float
    processing_time: float
    quality_score: int  # 1-5 scale
    relevance_score: int  # 1-5 scale
    completeness_score: int  # 1-5 scale
    comments: str
    evaluation_timestamp: str


class RAGEvaluator:
    """Comprehensive evaluator for RAG pipeline performance."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.evaluation_questions = self._create_evaluation_questions()
        
    def _create_evaluation_questions(self) -> List[EvaluationQuestion]:
        """Create a comprehensive set of evaluation questions."""
        questions = [
            # Easy questions - basic information retrieval
            EvaluationQuestion(
                question="What are the most common credit card billing issues?",
                expected_topics=["billing", "credit card", "charges", "fees"],
                expected_sources=3,
                difficulty="easy",
                category="credit_cards"
            ),
            EvaluationQuestion(
                question="Which financial products have the most complaints?",
                expected_topics=["products", "complaints", "volume", "categories"],
                expected_sources=2,
                difficulty="easy",
                category="general"
            ),
            
            # Medium questions - analysis and patterns
            EvaluationQuestion(
                question="What are the main causes of unauthorized charges on accounts?",
                expected_topics=["unauthorized", "charges", "security", "fraud"],
                expected_sources=4,
                difficulty="medium",
                category="security"
            ),
            EvaluationQuestion(
                question="How do billing disputes typically get resolved?",
                expected_topics=["disputes", "resolution", "process", "outcomes"],
                expected_sources=3,
                difficulty="medium",
                category="disputes"
            ),
            
            # Hard questions - complex analysis
            EvaluationQuestion(
                question="What are the risk factors that lead to account closures?",
                expected_topics=["risk", "account closure", "patterns", "triggers"],
                expected_sources=5,
                difficulty="hard",
                category="risk_analysis"
            ),
            EvaluationQuestion(
                question="How do customer service issues correlate with product types?",
                expected_topics=["customer service", "correlation", "product types", "patterns"],
                expected_sources=4,
                difficulty="hard",
                category="correlation_analysis"
            ),
            
            # Specific product questions
            EvaluationQuestion(
                question="What are the unique challenges with mortgage servicing?",
                expected_topics=["mortgage", "servicing", "challenges", "specifics"],
                expected_sources=3,
                difficulty="medium",
                category="mortgages"
            ),
            EvaluationQuestion(
                question="How do payday loan complaints differ from other loan types?",
                expected_topics=["payday loans", "differences", "comparison", "characteristics"],
                expected_sources=4,
                difficulty="medium",
                category="loans"
            ),
            
            # Temporal and trend questions
            EvaluationQuestion(
                question="Are there seasonal patterns in complaint volumes?",
                expected_topics=["seasonal", "patterns", "trends", "timing"],
                expected_sources=3,
                difficulty="hard",
                category="trends"
            ),
            EvaluationQuestion(
                question="What types of complaints have increased over time?",
                expected_topics=["trends", "increase", "patterns", "changes"],
                expected_sources=4,
                difficulty="medium",
                category="trends"
            )
        ]
        
        return questions
    
    def _assess_answer_quality(self, 
                              question: str, 
                              answer: str, 
                              expected_topics: List[str],
                              retrieved_sources: List[Dict]) -> Tuple[int, int, int]:
        """Assess the quality of the generated answer."""
        
        # Quality Score (1-5): Overall assessment
        quality_score = 3  # Start with neutral score
        
        # Relevance Score (1-5): How well the answer addresses the question
        relevance_score = 3
        
        # Completeness Score (1-5): How comprehensive the answer is
        completeness_score = 3
        
        # Check if answer contains expected topics
        answer_lower = answer.lower()
        topic_matches = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
        topic_coverage = topic_matches / len(expected_topics) if expected_topics else 0
        
        # Adjust scores based on topic coverage
        if topic_coverage >= 0.8:
            relevance_score = 5
            quality_score = 5
        elif topic_coverage >= 0.6:
            relevance_score = 4
            quality_score = 4
        elif topic_coverage >= 0.4:
            relevance_score = 3
            quality_score = 3
        elif topic_coverage >= 0.2:
            relevance_score = 2
            quality_score = 2
        else:
            relevance_score = 1
            quality_score = 1
        
        # Adjust completeness based on answer length and detail
        if len(answer) > 200 and "Don't have enough information" not in answer.lower():
            completeness_score = min(5, completeness_score + 1)
        elif len(answer) < 100:
            completeness_score = max(1, completeness_score - 1)
        
        # Adjust based on number of sources used
        if len(retrieved_sources) >= 3:
            completeness_score = min(5, completeness_score + 1)
        elif len(retrieved_sources) < 2:
            completeness_score = max(1, completeness_score - 1)
        
        # Check for specific details and examples
        if any(word in answer_lower for word in ["example", "specific", "case", "instance"]):
            completeness_score = min(5, completeness_score + 1)
        
        # Check for professional tone and structure
        if "based on the complaint data" in answer_lower or "the information shows" in answer_lower:
            quality_score = min(5, quality_score + 1)
        
        return quality_score, relevance_score, completeness_score
    
    def _generate_comments(self, 
                          question: str, 
                          answer: str, 
                          quality_score: int,
                          relevance_score: int,
                          completeness_score: int,
                          retrieved_sources: List[Dict]) -> str:
        """Generate detailed comments about the evaluation."""
        
        comments = []
        
        # Overall assessment
        if quality_score >= 4:
            comments.append("Excellent response with high relevance and completeness.")
        elif quality_score >= 3:
            comments.append("Good response with room for improvement in some areas.")
        else:
            comments.append("Response needs significant improvement in quality and relevance.")
        
        # Relevance feedback
        if relevance_score >= 4:
            comments.append("Answer directly addresses the question effectively.")
        elif relevance_score <= 2:
            comments.append("Answer shows limited relevance to the question.")
        
        # Completeness feedback
        if completeness_score >= 4:
            comments.append("Answer provides comprehensive coverage of the topic.")
        elif completeness_score <= 2:
            comments.append("Answer lacks depth and detail.")
        
        # Source utilization feedback
        if len(retrieved_sources) >= 4:
            comments.append("Good utilization of multiple sources for comprehensive coverage.")
        elif len(retrieved_sources) <= 2:
            comments.append("Limited source utilization may affect answer completeness.")
        
        # Specific improvements
        if "don't have enough information" in answer.lower():
            comments.append("System correctly identified information limitations.")
        
        if len(answer) < 150:
            comments.append("Answer could benefit from more detailed explanations.")
        
        return " ".join(comments)
    
    def evaluate_single_question(self, eval_question: EvaluationQuestion) -> EvaluationResult:
        """Evaluate the RAG pipeline on a single question."""
        
        logger.info(f"Evaluating question: {eval_question.question[:50]}...")
        
        try:
            # Run RAG pipeline
            start_time = time.time()
            response = self.rag_pipeline.run(eval_question.question, k=eval_question.expected_sources)
            processing_time = response.processing_time
            
            # Assess answer quality
            quality_score, relevance_score, completeness_score = self._assess_answer_quality(
                eval_question.question,
                response.answer,
                eval_question.expected_topics,
                response.retrieved_sources
            )
            
            # Generate comments
            comments = self._generate_comments(
                eval_question.question,
                response.answer,
                quality_score,
                relevance_score,
                completeness_score,
                response.retrieved_sources
            )
            
            # Prepare retrieved sources for evaluation
            retrieved_sources = []
            for source in response.retrieved_sources:
                retrieved_sources.append({
                    'text': source.text[:200] + "..." if len(source.text) > 200 else source.text,
                    'score': source.score,
                    'category': source.metadata.get('category', 'Unknown'),
                    'product': source.metadata.get('product', 'Unknown')
                })
            
            # Create evaluation result
            eval_result = EvaluationResult(
                question=eval_question.question,
                generated_answer=response.answer,
                retrieved_sources=retrieved_sources,
                confidence_score=response.confidence_score,
                processing_time=processing_time,
                quality_score=quality_score,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                comments=comments,
                evaluation_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Evaluation completed - Quality: {quality_score}, Relevance: {relevance_score}, Completeness: {completeness_score}")
            
            return eval_result
            
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
            
            # Return error result
            return EvaluationResult(
                question=eval_question.question,
                generated_answer=f"Error during evaluation: {str(e)}",
                retrieved_sources=[],
                confidence_score=0.0,
                processing_time=0.0,
                quality_score=1,
                relevance_score=1,
                completeness_score=1,
                comments=f"Evaluation failed due to error: {str(e)}",
                evaluation_timestamp=datetime.now().isoformat()
            )
    
    def run_full_evaluation(self) -> List[EvaluationResult]:
        """Run evaluation on all questions."""
        
        logger.info(f"Starting full evaluation with {len(self.evaluation_questions)} questions...")
        
        results = []
        for i, question in enumerate(self.evaluation_questions, 1):
            logger.info(f"Progress: {i}/{len(self.evaluation_questions)}")
            result = self.evaluate_single_question(question)
            results.append(result)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        logger.info("Full evaluation completed successfully!")
        return results
    
    def generate_evaluation_report(self, results: List[EvaluationResult]) -> str:
        """Generate a comprehensive evaluation report in Markdown format."""
        
        # Calculate summary statistics
        total_questions = len(results)
        avg_quality = sum(r.quality_score for r in results) / total_questions
        avg_relevance = sum(r.relevance_score for r in results) / total_questions
        avg_completeness = sum(r.completeness_score for r in results) / total_questions
        avg_confidence = sum(r.confidence_score for r in results) / total_questions
        avg_processing_time = sum(r.processing_time for r in results) / total_questions
        
        # Count scores by category
        quality_distribution = {}
        for i in range(1, 6):
            quality_distribution[i] = sum(1 for r in results if r.quality_score == i)
        
        # Generate markdown report
        report = f"""# RAG Pipeline Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Questions Evaluated:** {total_questions}

## Executive Summary

The RAG pipeline was evaluated on {total_questions} representative questions covering various aspects of financial complaint analysis.

### Overall Performance Metrics

| Metric | Score |
|--------|-------|
| **Average Quality Score** | {avg_quality:.2f}/5.00 |
| **Average Relevance Score** | {avg_relevance:.2f}/5.00 |
| **Average Completeness Score** | {avg_completeness:.2f}/5.00 |
| **Average Confidence Score** | {avg_confidence:.3f} |
| **Average Processing Time** | {avg_processing_time:.2f}s |

### Quality Score Distribution

| Score | Count | Percentage |
|-------|-------|------------|
"""
        
        for score in range(5, 0, -1):
            count = quality_distribution.get(score, 0)
            percentage = (count / total_questions) * 100
            report += f"| {score}/5 | {count} | {percentage:.1f}% |\n"
        
        report += f"""
## Detailed Evaluation Results

### Question-by-Question Analysis

"""
        
        for i, result in enumerate(results, 1):
            report += f"""#### Question {i}: {result.question}

**Generated Answer:** {result.generated_answer}

**Retrieved Sources ({len(result.retrieved_sources)}):**
"""
            
            for j, source in enumerate(result.retrieved_sources, 1):
                report += f"""- **Source {j}** (Score: {source['score']:.3f})
  - Category: {source['category']}
  - Product: {source['product']}
  - Text: {source['text']}

"""
            
            report += f"""**Evaluation Scores:**
- Quality: {result.quality_score}/5
- Relevance: {result.relevance_score}/5  
- Completeness: {result.completeness_score}/5
- Confidence: {result.confidence_score:.3f}
- Processing Time: {result.processing_time:.2f}s

**Comments:** {result.comments}

---
"""
        
        # Add recommendations section
        report += f"""
## Recommendations for Improvement

### Strengths
- The pipeline demonstrates good retrieval capabilities with an average confidence score of {avg_confidence:.3f}
- Processing times are reasonable at {avg_processing_time:.2f} seconds per query
- The system shows good understanding of financial complaint domains

### Areas for Improvement
"""
        
        if avg_quality < 3.5:
            report += "- **Answer Quality**: Focus on improving the relevance and completeness of generated responses\n"
        if avg_relevance < 3.5:
            report += "- **Relevance**: Ensure answers directly address the specific questions asked\n"
        if avg_completeness < 3.5:
            report += "- **Completeness**: Provide more detailed and comprehensive responses\n"
        
        report += """
### Technical Improvements
- Consider fine-tuning the prompt engineering for better answer quality
- Implement answer validation and fact-checking mechanisms
- Add support for multi-turn conversations
- Enhance source diversity in retrieval

### Business Impact
- Improved answer quality will lead to better customer service
- Faster processing times will enhance user experience
- Better source utilization will increase confidence in responses

## Conclusion

The RAG pipeline shows promise in financial complaint analysis with room for improvement in answer quality and relevance. The modular architecture allows for targeted enhancements in specific areas.
"""
        
        return report
    
    def export_results_to_csv(self, results: List[EvaluationResult], filename: str) -> None:
        """Export evaluation results to CSV format."""
        
        # Prepare data for CSV export
        export_data = []
        for result in results:
            # Flatten the result for CSV
            row = {
                'question': result.question,
                'generated_answer': result.generated_answer,
                'quality_score': result.quality_score,
                'relevance_score': result.relevance_score,
                'completeness_score': result.completeness_score,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'num_sources': len(result.retrieved_sources),
                'comments': result.comments,
                'timestamp': result.evaluation_timestamp
            }
            
            # Add source information
            for i, source in enumerate(result.retrieved_sources):
                row[f'source_{i+1}_category'] = source.get('category', 'Unknown')
                row[f'source_{i+1}_product'] = source.get('product', 'Unknown')
                row[f'source_{i+1}_score'] = source.get('score', 0.0)
            
            export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")
    
    def save_evaluation_report(self, results: List[EvaluationResult], 
                             report_dir: str = "../reports") -> str:
        """Save the evaluation report to a file."""
        
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report
        report_content = self.generate_evaluation_report(results)
        
        # Save markdown report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"rag_evaluation_report_{timestamp}.md"
        report_path = os.path.join(report_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save CSV results
        csv_filename = f"rag_evaluation_results_{timestamp}.csv"
        csv_path = os.path.join(report_dir, csv_filename)
        self.export_results_to_csv(results, csv_path)
        
        logger.info(f"Evaluation report saved to {report_path}")
        logger.info(f"Results CSV saved to {csv_path}")
        
        return report_path


def create_evaluator(vector_store_dir: str = "../vector_store") -> RAGEvaluator:
    """Convenience function to create an evaluator instance."""
    from .rag_pipeline import create_simple_pipeline
    
    pipeline = create_simple_pipeline(vector_store_dir)
    return RAGEvaluator(pipeline)


if __name__ == "__main__":
    # Example usage
    print("Testing RAG Evaluator...")
    
    try:
        # Create evaluator
        evaluator = create_evaluator()
        
        # Run evaluation on first few questions
        test_questions = evaluator.evaluation_questions[:3]
        print(f"Testing with {len(test_questions)} questions...")
        
        results = []
        for question in test_questions:
            result = evaluator.evaluate_single_question(question)
            results.append(result)
            print(f"Question: {question.question[:50]}...")
            print(f"Quality Score: {result.quality_score}/5")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print("-" * 50)
        
        # Generate and save report
        report_path = evaluator.save_evaluation_report(results)
        print(f"\nEvaluation report saved to: {report_path}")
        
        print("\nRAG Evaluator test completed successfully!")
        
    except Exception as e:
        print(f"Error testing RAG Evaluator: {e}")
