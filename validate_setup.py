#!/usr/bin/env python3
"""
Simple validation script to test basic functionality without requiring datasets.
This can be run locally to verify the code works before moving to cloud platforms.
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_file_structure():
    """Check if all required files exist."""
    required_files = [
        "README.md",
        "requirements.txt", 
        "setup.py",
        "models/__init__.py",
        "models/cnn_models.py",
        "models/vit_models.py",
        "models/hybrid_models.py",
        "data/data_preprocessing.py",
        "training/train_cnn.py",
        "training/train_vit.py",
        "training/train_hybrid.py",
        "evaluation/evaluate_models.py",
        "tests/test_models.py",
        "tests/test_data_loading.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        print("✅ All required files present")
        return True
    else:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

def check_syntax():
    """Check Python syntax of key files."""
    python_files = [
        "models/cnn_models.py",
        "models/vit_models.py", 
        "models/hybrid_models.py",
        "data/data_preprocessing.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ {file_path} - Syntax OK")
        except SyntaxError as e:
            syntax_errors.append((file_path, str(e)))
            print(f"❌ {file_path} - Syntax Error: {e}")
        except FileNotFoundError:
            print(f"⚠️  {file_path} - File not found")
    
    return len(syntax_errors) == 0

def check_imports():
    """Check if we can import our modules (without external dependencies)."""
    try:
        # Test if our modules can be imported structurally
        spec = importlib.util.spec_from_file_location("models", "models/__init__.py")
        if spec is None:
            print("❌ Cannot load models module")
            return False
        print("✅ Models module structure OK")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_dependencies():
    """Check if key dependencies are available."""
    dependencies = [
        "torch", "torchvision", "transformers", "timm", 
        "numpy", "pandas", "matplotlib", "seaborn", 
        "scikit-learn", "Pillow", "tqdm"
    ]
    
    available = []
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            available.append(dep)
        except ImportError:
            missing.append(dep)
    
    print(f"✅ Available dependencies ({len(available)}): {', '.join(available[:5])}...")
    if missing:
        print(f"⚠️  Missing dependencies ({len(missing)}): {', '.join(missing[:5])}...")
        print("   Run: pip install -r requirements.txt")
    
    return len(missing) == 0

def main():
    """Run all validation checks."""
    print("🔍 Validating Plant Disease Detection Project Setup")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("File Structure", check_file_structure), 
        ("Python Syntax", check_syntax),
        ("Module Structure", check_imports),
        ("Dependencies", check_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, _) in enumerate(checks):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Project is ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download PlantVillage dataset")
        print("3. Run: python data/data_preprocessing.py")
        print("4. Start training: python training/train_cnn.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
