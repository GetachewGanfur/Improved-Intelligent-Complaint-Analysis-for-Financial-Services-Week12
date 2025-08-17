"""
Configuration settings for the complaint analysis system 
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Data files
RAW_DATA_FILE = DATA_DIR / "complaints.csv"
FILTERED_DATA_FILE = DATA_DIR / "filtered_complaints.csv"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Target products for analysis
TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Buy now pay later", 
    "Savings account",
    "Money transfers"
]

# RAG settings
TOP_K_RETRIEVAL = 5
MAX_CONTEXT_LENGTH = 2000

# UI settings
APP_TITLE = "Intelligent Financial Complaint Analysis"
APP_ICON = "💰"