#!/usr/bin/env python3
"""
Comprehensive Application Launcher for Financial Complaint Analysis RAG System

This enhanced launcher provides multiple modes of operation:
- Streamlit web interface
- Command-line RAG pipeline
- System evaluation and testing
- Data preprocessing and setup
"""

import subprocess
import sys
import os
import argparse
import json
from pathlib import Path

# Suppress Streamlit ScriptRunContext warning when running outside Streamlit
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

def check_dependencies():
    """Check and install required dependencies."""
    required_packages = {
        'streamlit': 'streamlit>=1.28.0',
        'pandas': 'pandas>=1.5.0',
        'sentence_transformers': 'sentence-transformers>=2.2.0',
        'faiss': 'faiss-cpu>=1.7.0'
    }
    
    missing_packages = []
    
    for package, requirement in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(requirement)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        response = input("Install missing packages? (y/n): ").lower()
        if response == 'y':
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("All packages installed successfully!")
        else:
            print("Some features may not work without required packages.")
    else:
        print("All required packages are installed")

def check_system_status():
    """Check the status of the RAG system components."""
    status = {
        'data_file': os.path.exists('data/complaints.csv'),
        'vector_store': os.path.exists('vector_store'),
        'src_directory': os.path.exists('src'),
        'config_file': os.path.exists('config.py')
    }
    
    print("\nSystem Status:")
    print("=" * 40)
    for component, exists in status.items():
        status_text = "Ready" if exists else "Missing"
        print(f"{component.replace('_', ' ').title()}: {status_text}")
    
    return all(status.values())

def setup_data():
    """Setup and preprocess data if needed."""
    if not os.path.exists('data/complaints.csv'):
        print("Raw data file not found: data/complaints.csv")
        print("Please ensure the complaints dataset is available.")
        return False
    
    if not os.path.exists('vector_store'):
        print("Vector store not found. Creating...")
        try:
            # Import and run the embedding creation
            sys.path.append('src')
            from embedding_indexer import create_vector_store
            create_vector_store()
            print("Vector store created successfully!")
        except Exception as e:
            print(f"Failed to create vector store: {e}")
            print("Please run Task 2 notebook manually:")
            print("   jupyter notebook notebooks/02_text_chunking_embedding_vectorstore.ipynb")
            return False
    
    return True

def launch_streamlit():
    """Launch the Streamlit web interface."""
    print("Launching Streamlit Web Interface...")
    print("=" * 60)
    
    try:
        import streamlit
        print(f"Streamlit {streamlit.__version__} is ready")
    except ImportError:
        print("Streamlit not available. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"])
    
    print("\nStarting web interface...")
    print("The app will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the application.")
    print("=" * 60)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/ui_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError launching Streamlit: {e}")

def launch_cli():
    """Launch the command-line interface."""
    print("Launching Command-Line Interface...")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "src/main.py", "--interactive"])
    except Exception as e:
        print(f"Error launching CLI: {e}")

def run_evaluation():
    """Run the RAG system evaluation."""
    print("Running RAG System Evaluation...")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "src/main.py", "--evaluate"])
    except Exception as e:
        print(f"Error running evaluation: {e}")

def run_tests():
    """Run system tests and diagnostics."""
    print("Running System Tests...")
    print("=" * 60)
    
    # Test RAG pipeline
    try:
        sys.path.append('src')
        from rag_pipeline import create_simple_pipeline
        
        pipeline = create_simple_pipeline()
        test_query = "What are common credit card issues?"
        response = pipeline.run(test_query)
        
        print("RAG Pipeline Test: PASSED")
        print(f"   Query: {test_query}")
        print(f"   Response length: {len(response.answer)} chars")
        print(f"   Sources retrieved: {len(response.retrieved_sources)}")
        print(f"   Confidence: {response.confidence_score:.3f}")
        
    except Exception as e:
        print(f"RAG Pipeline Test: FAILED - {e}")
    
    # Test vector store
    try:
        from vector_store_utils import ComplaintVectorStore
        
        vs = ComplaintVectorStore("vector_store")
        vs.load()
        stats = vs.get_stats()
        
        print("Vector Store Test: PASSED")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Embedding dimension: {stats['embedding_dim']}")
        
    except Exception as e:
        print(f"Vector Store Test: FAILED - {e}")

def show_system_info():
    """Display comprehensive system information."""
    print("Financial Complaint Analysis RAG System")
    print("=" * 60)
    print("\nAvailable Modes:")
    print("   web      - Launch Streamlit web interface (default)")
    print("   cli      - Launch command-line interface")
    print("   eval     - Run RAG system evaluation")
    print("   test     - Run system tests and diagnostics")
    print("   setup    - Setup and preprocess data")
    print("   status   - Check system status")
    print("   info     - Show this information")
    
    print("\nExample Usage:")
    print("   python app.py web          # Launch web interface")
    print("   python app.py cli          # Launch CLI")
    print("   python app.py eval         # Run evaluation")
    print("   python app.py test         # Run tests")
    print("   python app.py setup        # Setup system")
    
    print("\nKey Features:")
    print("   - Interactive chat interface for complaint analysis")
    print("   - Semantic search across financial complaint data")
    print("   - AI-powered answer generation")
    print("   - Comprehensive evaluation framework")
    print("   - Source transparency and metadata display")
    
    print("\nSystem Requirements:")
    print("   - Python 3.8+")
    print("   - 4GB+ RAM (for embeddings)")
    print("   - Internet connection (for model downloads)")
    print("   - ~2GB disk space (for models and data)")

def main():
    """Main application launcher with multiple modes."""
    parser = argparse.ArgumentParser(
        description="Financial Complaint Analysis RAG System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        default='web',
        choices=['web', 'cli', 'eval', 'test', 'setup', 'status', 'info'],
        help='Launch mode (default: web)'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check and install dependencies'
    )
    
    parser.add_argument(
        '--force-setup',
        action='store_true',
        help='Force data setup even if exists'
    )
    
    args = parser.parse_args()
    
    # Show header
    print("Financial Complaint Analysis RAG System")
    print("=" * 60)
    
    # Check dependencies if requested
    if args.check_deps:
        check_dependencies()
    
    # Handle different modes
    if args.mode == 'info':
        show_system_info()
        return
    
    elif args.mode == 'status':
        system_ready = check_system_status()
        if not system_ready:
            print("\nSystem not fully ready. Run 'python app.py setup' to initialize.")
        else:
            print("\nSystem is ready for use!")
        return
    
    elif args.mode == 'setup':
        print("Setting up system...")
        check_dependencies()
        if setup_data():
            print("System setup completed successfully!")
        else:
            print("System setup failed. Please check the logs above.")
        return
    
    elif args.mode == 'test':
        run_tests()
        return
    
    elif args.mode == 'eval':
        run_evaluation()
        return
    
    elif args.mode == 'cli':
        if not check_system_status():
            print("System not ready. Run 'python app.py setup' first.")
            return
        launch_cli()
        return
    
    elif args.mode == 'web':
        if not check_system_status():
            print("System not ready. Run 'python app.py setup' first.")
            return
        launch_streamlit()
        return
    
    else:
        print(f"Unknown mode: {args.mode}")
        show_system_info()

if __name__ == "__main__":
    main()