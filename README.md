#Improved Intelligent Complaint Analysis for Financial Services - Week 12

## Project Overview

This project implements a comprehensive RAG (Retrieval-Augmented Generation) system for analyzing financial service complaints. The system provides intelligent query answering based on complaint data through semantic search and AI-powered text generation.

## 🏗️ Architecture 

The system is built with a modular, object-oriented design:

```plaintext
src/
├── rag_pipeline.py          # Core RAG pipeline implementation
├── rag_evaluator.py         # Comprehensive evaluation framework
├── vector_store_utils.py    # Vector store management utilities
├── main.py                  # Command-line application interface
└── ...

notebooks/
├── 01_eda_preprocessing.ipynb           # Task 1: Data exploration and preprocessing
├── 02_text_chunking_embedding_vectorstore.ipynb  # Task 2: Vector store creation
└── 03_rag_pipeline_evaluation.ipynb    # Task 3: RAG implementation and evaluation
```

## 🚀 Features

### Core RAG Pipeline

*   **Semantic Retrieval**: FAISS-based vector search with sentence transformers
*   **Intelligent Prompting**: Domain-specific prompt engineering for financial complaints
*   **Flexible Generation**: Support for both Hugging Face models and mock generation
*   **Modular Design**: Easy to extend and customize components

### Evaluation Framework

*   **Comprehensive Assessment**: 10 predefined evaluation questions across difficulty levels
*   **Multi-dimensional Scoring**: Quality, relevance, and completeness metrics (1-5 scale)
*   **Automated Analysis**: Detailed reports with actionable insights
*   **Performance Visualization**: Charts and metrics for system optimization

### User Interface

*   **Interactive Mode**: Real-time query answering
*   **Command-line Tools**: Scriptable evaluation and testing
*   **Comprehensive Reporting**: Markdown and CSV export capabilities

## 📋 Prerequisites

*   Python 3.8+
*   Required packages (see `requirements.txt`):
    *   Core: pandas, numpy, matplotlib, seaborn
    *   NLP: sentence-transformers, transformers, torch
    *   Vector DB: faiss-cpu
    *   Utilities: langchain, tqdm

## 🛠️ Installation

**Clone the repository**:

**Install dependencies**:

**Verify installation**:

## 📖 Usage

### Quick Start

**Run Task 2 notebook first** to create the vector store:

**Test the RAG system**:

**Run Task 3 notebook** for full evaluation:

### Interactive Query Mode

```plaintext
# Start interactive session
python src/main.py --interactive

# Example queries:
# - What are the most common credit card billing issues?
# - How do customers resolve billing disputes?
# - What are the main security concerns with financial accounts?
```

### Evaluation Mode

```plaintext
# Run full evaluation
python src/main.py --evaluate

# Evaluate specific question
python src/main.py --evaluate-question "What are credit card fraud patterns?"

# Run evaluation without saving report
python src/main.py --evaluate --no-report
```

### Programmatic Usage

```python
from src.rag_pipeline import create_simple_pipeline
from src.rag_evaluator import create_evaluator

# Create RAG pipeline
pipeline = create_simple_pipeline('../vector_store')

# Ask questions
response = pipeline.run("What are common billing issues?")
print(response.answer)

# Run evaluation
evaluator = create_evaluator('../vector_store')
results = evaluator.run_full_evaluation()
```

## 🔍 System Components

### 1\. RAG Pipeline (`rag_pipeline.py`)

**Core Classes**:

*   `RAGPipeline`: Main orchestrator
*   `VectorStoreRetriever`: Semantic search implementation
*   `FinancialComplaintPromptEngine`: Domain-specific prompting
*   `MockGenerator`/`HuggingFaceGenerator`: Text generation

**Key Features**:

*   Configurable retrieval parameters (top-k, similarity thresholds)
*   Intelligent context preparation and truncation
*   Confidence scoring based on retrieval quality
*   Error handling and graceful degradation

### 2\. Evaluation Framework (`rag_evaluator.py`)

**Evaluation Questions**:

*   **Easy**: Basic information retrieval (2 questions)
*   **Medium**: Analysis and pattern recognition (4 questions)
*   **Hard**: Complex correlation and trend analysis (4 questions)

**Scoring Metrics**:

*   **Quality Score**: Overall assessment (1-5)
*   **Relevance Score**: Question-answer alignment (1-5)
*   **Completeness Score**: Answer comprehensiveness (1-5)
*   **Confidence Score**: Retrieval similarity (0-1)

### 3\. Vector Store Utilities (`vector_store_utils.py`)

**Features**:

*   FAISS index management
*   Metadata preservation and filtering
*   Category-based search capabilities
*   Export and statistics functionality

## 📊 Evaluation Results

The system provides comprehensive evaluation reports including:

*   **Performance Metrics**: Average scores across all dimensions
*   **Question Analysis**: Detailed breakdown of each evaluation question
*   **Source Utilization**: Analysis of retrieved document quality
*   **Recommendations**: Actionable insights for improvement
*   **Visualizations**: Performance charts and distributions

## 🎨 Interactive Chat Interface

### Features

*   **Real-time Chat**: Interactive question-answer interface
*   **Source Transparency**: Display retrieved text chunks with metadata
*   **Configurable Settings**: Adjustable retrieval parameters
*   **Chat History**: Conversation tracking and management
*   **Responsive Design**: Works on desktop, tablet, and mobile

### Launch Methods

```plaintext
# Method 1: Easy launcher (recommended)
python app.py

# Method 2: Direct Streamlit
streamlit run src/ui_app.py

# Method 3: Python module
python -m streamlit run src/ui_app.py
```

### UI Components

*   **ChatInterface**: Main application class
*   **Question Input**: Text area for user queries
*   **Response Display**: AI-generated answers with metrics
*   **Source Panel**: Expandable source information
*   **Configuration Sidebar**: Settings and example questions

## 🔧 Configuration

### RAG Pipeline Settings

```python
from src.rag_pipeline import RAGConfig

config = RAGConfig(
    top_k=5,                    # Number of documents to retrieve
    similarity_threshold=0.3,    # Minimum similarity score
    max_context_length=2000,    # Maximum context length
    temperature=0.7,            # Generation randomness
    max_tokens=500              # Maximum response length
)
```

### UI Configuration

```python
# Streamlit app configuration
streamlit_config = {
    'server.port': 8501,           # Default port
    'server.address': 'localhost',  # Server address
    'browser.gatherUsageStats': False  # Privacy setting
}
```

### Model Selection

```python
# Use mock generator for testing
pipeline = create_simple_pipeline(use_mock_generator=True)

# Use Hugging Face model
pipeline = create_simple_pipeline(
    model_name="microsoft/DialoGPT-medium"
)
```

## 📈 Performance Optimization

### Current Performance

*   **Processing Time**: Typically 1-3 seconds per query
*   **Retrieval Quality**: High similarity scores for relevant queries
*   **Answer Quality**: Good relevance with room for improvement in completeness

### Optimization Opportunities

1.  **Prompt Engineering**: Refine templates for better answer quality
2.  **Retrieval Parameters**: Tune similarity thresholds and chunk sizes
3.  **Model Fine-tuning**: Domain-specific training for financial complaints
4.  **Context Management**: Improve source selection and ranking

## 🧪 Testing

### Automated Tests

```plaintext
# Run basic functionality tests
python test_rag_system.py

# Test specific components
python -c "from src.rag_pipeline import MockGenerator; print('✅ Import successful')"
```

### Manual Testing

1.  **Interactive Mode**: Test real-time queries
2.  **Evaluation Mode**: Assess system performance
3.  **Custom Questions**: Test with domain-specific queries

## 📝 Output Files

The system generates several output files:

*   **Evaluation Reports**: Markdown format with detailed analysis
*   **Results CSV**: Structured data for further analysis
*   **Visualizations**: Performance charts and metrics
*   **Logs**: Detailed execution logs for debugging

## 🤝 Contributing

### Development Workflow

1.  **Fork** the repository
2.  **Create** a feature branch
3.  **Implement** changes with proper testing
4.  **Submit** a pull request with detailed description

### Code Standards

*   Follow PEP 8 style guidelines
*   Include comprehensive docstrings
*   Add unit tests for new functionality
*   Update documentation as needed

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:

**Vector Store Not Found**:

**Memory Issues**:

**Model Loading Errors**:

### Debug Mode

```plaintext
# Enable verbose logging
python src/main.py --verbose --interactive

# Check system status
python test_rag_system.py
```

## 📚 Additional Resources

### Documentation

*   [Sentence Transformers](https://www.sbert.net/)
*   [FAISS Documentation](https://github.com/facebookresearch/faiss)
*   [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Research Papers

*   "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.)
*   "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al.)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

*   Financial complaint data from CFPB
*   Sentence Transformers for semantic embeddings
*   FAISS for efficient similarity search
*   Hugging Face for transformer models

**Note**: This system is designed for educational and research purposes. For production use, ensure proper data privacy and security measures are in place.  
 

```plaintext
# Use mock generator for testing
python src/main.py --interactive
```

```plaintext
# Use CPU-only FAISS
pip install faiss-cpu
```

```plaintext
# Run Task 2 notebook first
jupyter notebook notebooks/02_text_chunking_embedding_vectorstore.ipynb
```

```plaintext
# Ensure src directory is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

```plaintext
jupyter notebook notebooks/03_rag_pipeline_evaluation.ipynb
```

```plaintext
python test_rag_system.py
```

```plaintext
jupyter notebook notebooks/02_text_chunking_embedding_vectorstore.ipynb
```

```plaintext
python test_rag_system.py
```

```plaintext
pip install -r requirements.txt
```

```plaintext
git clone <repository-url>
cd Intelligent-Complaint-Analysis-for-Financial-Services-Week6
```