"""
System Health Monitor for Financial Complaint Analysis RAG System

This module provides comprehensive system monitoring, health checks,
and performance diagnostics for the RAG pipeline.
"""

import os
import sys
import time
import json
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemHealth:
    """System health status information."""
    timestamp: str
    overall_status: str  # 'healthy', 'warning', 'critical'
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    vector_store_status: bool
    rag_pipeline_status: bool
    dependencies_status: bool
    data_integrity: bool
    performance_score: float
    issues: List[str]
    recommendations: List[str]


class SystemMonitor:
    """Comprehensive system monitoring and health checks."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.vector_store_dir = self.base_dir / "vector_store"
        self.data_dir = self.base_dir / "data"
        self.src_dir = self.base_dir / "src"
        
    def check_system_resources(self) -> Dict[str, float]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.base_dir))
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'memory_available_gb': 0.0,
                'disk_usage': 0.0,
                'disk_free_gb': 0.0
            }
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if all required dependencies are installed."""
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'sentence_transformers',
            'faiss', 'streamlit', 'transformers', 'torch'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                # Handle package name variations
                import_name = package.replace('-', '_')
                if package == 'faiss':
                    import_name = 'faiss'
                elif package == 'scikit-learn':
                    import_name = 'sklearn'
                
                __import__(import_name)
            except ImportError:
                missing_packages.append(package)
        
        return len(missing_packages) == 0, missing_packages
    
    def check_data_integrity(self) -> Tuple[bool, List[str]]:
        """Check data files and integrity."""
        issues = []
        
        # Check raw data file
        complaints_file = self.data_dir / "complaints.csv"
        if not complaints_file.exists():
            issues.append("Raw complaints data file missing")
        else:
            try:
                import pandas as pd
                df = pd.read_csv(complaints_file, nrows=100)  # Quick check
                if df.empty:
                    issues.append("Complaints data file is empty")
                elif 'Consumer complaint narrative' not in df.columns:
                    issues.append("Required columns missing in complaints data")
            except Exception as e:
                issues.append(f"Error reading complaints data: {str(e)}")
        
        # Check filtered data
        filtered_file = self.data_dir / "filtered_complaints.csv"
        if filtered_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(filtered_file, nrows=100)
                if df.empty:
                    issues.append("Filtered complaints data is empty")
            except Exception as e:
                issues.append(f"Error reading filtered data: {str(e)}")
        
        return len(issues) == 0, issues
    
    def check_vector_store(self) -> Tuple[bool, List[str]]:
        """Check vector store status and integrity."""
        issues = []
        
        if not self.vector_store_dir.exists():
            issues.append("Vector store directory missing")
            return False, issues
        
        # Check for required files
        required_files = ['index.faiss', 'metadata.json', 'chunks.json']
        for file_name in required_files:
            file_path = self.vector_store_dir / file_name
            if not file_path.exists():
                issues.append(f"Vector store file missing: {file_name}")
        
        # Try to load vector store
        try:
            sys.path.append(str(self.src_dir))
            from vector_store_utils import ComplaintVectorStore
            
            vs = ComplaintVectorStore(str(self.vector_store_dir))
            vs.load()
            stats = vs.get_stats()
            
            if stats['total_chunks'] == 0:
                issues.append("Vector store contains no chunks")
            elif stats['total_chunks'] < 1000:
                issues.append(f"Vector store has only {stats['total_chunks']} chunks (may be incomplete)")
                
        except Exception as e:
            issues.append(f"Error loading vector store: {str(e)}")
        
        return len(issues) == 0, issues
    
    def check_rag_pipeline(self) -> Tuple[bool, List[str]]:
        """Check RAG pipeline functionality."""
        issues = []
        
        try:
            sys.path.append(str(self.src_dir))
            from rag_pipeline import create_simple_pipeline
            
            # Test pipeline creation
            pipeline = create_simple_pipeline(str(self.vector_store_dir))
            
            # Test simple query
            test_query = "What are credit card issues?"
            start_time = time.time()
            response = pipeline.run(test_query)
            response_time = time.time() - start_time
            
            # Check response quality
            if not response.answer:
                issues.append("RAG pipeline returns empty responses")
            elif len(response.answer) < 10:
                issues.append("RAG pipeline responses are too short")
            
            if len(response.retrieved_sources) == 0:
                issues.append("RAG pipeline retrieves no sources")
            
            if response_time > 30:
                issues.append(f"RAG pipeline is slow (>{response_time:.1f}s)")
            
            if response.confidence_score < 0.1:
                issues.append("RAG pipeline has low confidence scores")
                
        except Exception as e:
            issues.append(f"Error testing RAG pipeline: {str(e)}")
        
        return len(issues) == 0, issues
    
    def calculate_performance_score(self, resources: Dict, issues: List[str]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        # Resource penalties
        if resources['cpu_usage'] > 80:
            score -= 20
        elif resources['cpu_usage'] > 60:
            score -= 10
        
        if resources['memory_usage'] > 90:
            score -= 25
        elif resources['memory_usage'] > 75:
            score -= 15
        
        if resources['disk_usage'] > 95:
            score -= 20
        elif resources['disk_usage'] > 85:
            score -= 10
        
        # Issue penalties
        score -= len(issues) * 5
        
        return max(0.0, score)
    
    def generate_recommendations(self, health: SystemHealth) -> List[str]:
        """Generate actionable recommendations based on health status."""
        recommendations = []
        
        # Resource recommendations
        if health.cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider closing other applications.")
        
        if health.memory_usage > 75:
            recommendations.append("High memory usage. Consider reducing top-k retrieval or using smaller models.")
        
        if health.disk_usage > 85:
            recommendations.append("Low disk space. Consider cleaning up temporary files or logs.")
        
        # System-specific recommendations
        if not health.vector_store_status:
            recommendations.append("Run Task 2 notebook to create the vector store.")
        
        if not health.rag_pipeline_status:
            recommendations.append("Check RAG pipeline configuration and dependencies.")
        
        if not health.dependencies_status:
            recommendations.append("Install missing dependencies: pip install -r requirements.txt")
        
        if not health.data_integrity:
            recommendations.append("Verify data files are complete and accessible.")
        
        if health.performance_score < 50:
            recommendations.append("System performance is degraded. Consider restarting or checking logs.")
        
        return recommendations
    
    def run_health_check(self) -> SystemHealth:
        """Run comprehensive health check."""
        logger.info("Running system health check...")
        
        # Check system resources
        resources = self.check_system_resources()
        
        # Check components
        deps_ok, missing_deps = self.check_dependencies()
        data_ok, data_issues = self.check_data_integrity()
        vs_ok, vs_issues = self.check_vector_store()
        rag_ok, rag_issues = self.check_rag_pipeline()
        
        # Collect all issues
        all_issues = []
        if missing_deps:
            all_issues.extend([f"Missing dependency: {dep}" for dep in missing_deps])\n        all_issues.extend(data_issues)\n        all_issues.extend(vs_issues)\n        all_issues.extend(rag_issues)\n        \n        # Calculate performance score\n        performance_score = self.calculate_performance_score(resources, all_issues)\n        \n        # Determine overall status\n        if len(all_issues) == 0 and performance_score > 80:\n            overall_status = \"healthy\"\n        elif len(all_issues) <= 2 and performance_score > 60:\n            overall_status = \"warning\"\n        else:\n            overall_status = \"critical\"\n        \n        # Create health object\n        health = SystemHealth(\n            timestamp=datetime.now().isoformat(),\n            overall_status=overall_status,\n            cpu_usage=resources['cpu_usage'],\n            memory_usage=resources['memory_usage'],\n            disk_usage=resources['disk_usage'],\n            vector_store_status=vs_ok,\n            rag_pipeline_status=rag_ok,\n            dependencies_status=deps_ok,\n            data_integrity=data_ok,\n            performance_score=performance_score,\n            issues=all_issues,\n            recommendations=[]\n        )\n        \n        # Generate recommendations\n        health.recommendations = self.generate_recommendations(health)\n        \n        logger.info(f\"Health check completed. Status: {overall_status}\")\n        return health
    
    def save_health_report(self, health: SystemHealth, filepath: Optional[str] = None) -> str:
        """Save health report to file."""
        if filepath is None:\n            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n            filepath = f\"health_report_{timestamp}.json\"\n        \n        with open(filepath, 'w') as f:\n            json.dump(asdict(health), f, indent=2)\n        \n        logger.info(f\"Health report saved to: {filepath}\")\n        return filepath
    
    def display_health_summary(self, health: SystemHealth):\n        \"\"\"Display a formatted health summary.\"\"\"\n        status_icons = {\n            \"healthy\": \"🟢\",\n            \"warning\": \"🟡\",\n            \"critical\": \"🔴\"\n        }\n        \n        print(\"\\n\" + \"=\" * 60)\n        print(f\"🏥 SYSTEM HEALTH REPORT - {health.timestamp}\")\n        print(\"=\" * 60)\n        \n        # Overall status\n        icon = status_icons.get(health.overall_status, \"❓\")\n        print(f\"\\n{icon} Overall Status: {health.overall_status.upper()}\")\n        print(f\"📊 Performance Score: {health.performance_score:.1f}/100\")\n        \n        # Resource usage\n        print(\"\\n💻 System Resources:\")\n        print(f\"   CPU Usage: {health.cpu_usage:.1f}%\")\n        print(f\"   Memory Usage: {health.memory_usage:.1f}%\")\n        print(f\"   Disk Usage: {health.disk_usage:.1f}%\")\n        \n        # Component status\n        print(\"\\n🔧 Component Status:\")\n        components = [\n            (\"Dependencies\", health.dependencies_status),\n            (\"Data Integrity\", health.data_integrity),\n            (\"Vector Store\", health.vector_store_status),\n            (\"RAG Pipeline\", health.rag_pipeline_status)\n        ]\n        \n        for name, status in components:\n            icon = \"✅\" if status else \"❌\"\n            print(f\"   {icon} {name}: {'OK' if status else 'FAILED'}\")\n        \n        # Issues\n        if health.issues:\n            print(\"\\n⚠️  Issues Found:\")\n            for i, issue in enumerate(health.issues, 1):\n                print(f\"   {i}. {issue}\")\n        \n        # Recommendations\n        if health.recommendations:\n            print(\"\\n💡 Recommendations:\")\n            for i, rec in enumerate(health.recommendations, 1):\n                print(f\"   {i}. {rec}\")\n        \n        print(\"\\n\" + \"=\" * 60)\n\n\ndef main():\n    \"\"\"Run system health check from command line.\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description=\"System Health Monitor\")\n    parser.add_argument('--save', action='store_true', help='Save report to file')\n    parser.add_argument('--output', help='Output file path')\n    parser.add_argument('--quiet', action='store_true', help='Minimal output')\n    \n    args = parser.parse_args()\n    \n    # Run health check\n    monitor = SystemMonitor()\n    health = monitor.run_health_check()\n    \n    # Display results\n    if not args.quiet:\n        monitor.display_health_summary(health)\n    \n    # Save report if requested\n    if args.save:\n        filepath = monitor.save_health_report(health, args.output)\n        print(f\"\\n📄 Report saved to: {filepath}\")\n    \n    # Exit with appropriate code\n    if health.overall_status == \"critical\":\n        sys.exit(1)\n    elif health.overall_status == \"warning\":\n        sys.exit(2)\n    else:\n        sys.exit(0)\n\n\nif __name__ == \"__main__\":\n    main()\n