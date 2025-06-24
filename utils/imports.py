"""
Centralized import utilities to avoid path issues
"""

import os
import sys
from pathlib import Path

def add_project_root():
    """Add project root to Python path"""
    # Find project root (where budget.py is located)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Look for budget.py as marker of project root
    while project_root != project_root.parent:
        if (project_root / "budget.py").exists():
            break
        project_root = project_root.parent
    
    # Add to path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root

def safe_import_budget():
    """Safely import budget tracker"""
    try:
        add_project_root()
        from budget import BudgetTracker
        return BudgetTracker
    except ImportError:
        # Return a dummy class if budget.py not found
        class DummyBudgetTracker:
            def record_expense(self, hours, description):
                print(f"Budget tracking unavailable: {description} - {hours} hours")
            def get_summary(self):
                return "Budget tracking unavailable"
        return DummyBudgetTracker