#!/usr/bin/env python3
"""
Test script to validate all fixes are working
"""

import os
import sys
from pathlib import Path
import importlib.util

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    # Test utils imports
    try:
        from utils.imports import add_project_root, safe_import_budget
        print("✓ utils.imports working")
    except ImportError as e:
        print(f"✗ utils.imports failed: {e}")
        return False
    
    # Test budget import
    try:
        BudgetTracker = safe_import_budget()
        tracker = BudgetTracker()
        print("✓ Budget tracking working")
    except Exception as e:
        print(f"✗ Budget tracking failed: {e}")
    
    return True

def test_experiment_files():
    """Test that all experiment files exist"""
    print("Testing experiment files...")
    
    required_files = [
        "experiments/exp3-dataset-mixing/data_mixer.py",
        "experiments/exp4-zero-cost-eval/evaluator.py",
        "utils/imports.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing")
            all_good = False
    
    return all_good

def test_module_imports():
    """Test that modules can be imported"""
    print("Testing module imports...")
    
    # Test exp3 import
    try:
        sys.path.append("experiments/exp3-dataset-mixing")
        import data_mixer
        print("✓ exp3 data_mixer imports correctly")
    except ImportError as e:
        print(f"✗ exp3 data_mixer import failed: {e}")
        return False
    
    # Test exp4 import
    try:
        sys.path.append("experiments/exp4-zero-cost-eval")
        import evaluator
        print("✓ exp4 evaluator imports correctly")
    except ImportError as e:
        print(f"✗ exp4 evaluator import failed: {e}")
        return False
    
    return True

def main():
    print("Running validation tests for all fixes...")
    print("=" * 50)
    
    tests = [
        ("Import structure", test_imports),
        ("File existence", test_experiment_files),
        ("Module imports", test_module_imports)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! Ready to run experiments.")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())