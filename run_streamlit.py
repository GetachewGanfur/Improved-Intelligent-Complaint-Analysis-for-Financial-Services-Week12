#!/usr/bin/env python3
"""
Simple launcher for the Streamlit RAG application
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    print("Launching CrediTrust Complaint Analysis Assistant...")
    print("=" * 60)
    
    # Path to the Streamlit app
    app_path = Path(__file__).parent / "src" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        return 1
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("Starting Streamlit server...")
        print("The app will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop the application.")
        print("=" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
        return 0
    except Exception as e:
        print(f"\nError launching Streamlit: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())