"""
Verify that the code package is complete and ready for release.

This script checks:
- All required files present
- Python syntax valid
- Dependencies importable
- No sensitive data
- Documentation complete
"""

import sys
from pathlib import Path
import ast
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def check(condition, message):
    """Print check result."""
    if condition:
        print(f"{GREEN}✓{RESET} {message}")
        return True
    else:
        print(f"{RED}✗{RESET} {message}")
        return False


def warn(message):
    """Print warning."""
    print(f"{YELLOW}⚠{RESET} {message}")


def main():
    print("="*80)
    print("Package Verification")
    print("="*80)

    root = Path(__file__).parent.parent
    passed = 0
    failed = 0

    # 1. Check required files
    print("\n1. Required Files")
    required_files = [
        'README.md',
        'LICENSE',
        'CITATION.cff',
        'requirements.txt',
        'setup.py',
        '.gitignore',
        'CHANGELOG.md',
    ]

    for file in required_files:
        path = root / file
        if check(path.exists(), f"{file}"):
            passed += 1
        else:
            failed += 1

    # 2. Check source code
    print("\n2. Source Code")
    src_dir = root / "src" / "eqgrid"

    if check(src_dir.exists(), "src/eqgrid/ directory"):
        passed += 1

        # Check key modules
        key_modules = [
            'download.py',
            'build_tensors.py',
            'train.py',
            'evaluation.py',
            'baselines.py',
            'models/convlstm.py',
        ]

        for module in key_modules:
            path = src_dir / module
            if check(path.exists(), f"  {module}"):
                passed += 1

                # Check Python syntax
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                    passed += 1
                except SyntaxError as e:
                    failed += 1
                    print(f"    {RED}Syntax error: {e}{RESET}")
            else:
                failed += 1
    else:
        failed += 1

    # 3. Check scripts
    print("\n3. Scripts")
    scripts_dir = root / "scripts"

    if check(scripts_dir.exists(), "scripts/ directory"):
        passed += 1

        key_scripts = [
            'fix_paper_metrics_unified.py',
            'run_comprehensive_evaluation.py',
        ]

        for script in key_scripts:
            path = scripts_dir / script
            if check(path.exists(), f"  {script}"):
                passed += 1
            else:
                failed += 1
    else:
        failed += 1

    # 4. Check documentation
    print("\n4. Documentation")
    docs_to_check = [
        ('README.md', ['Installation', 'Quick Start', 'Citation']),
        ('docs/REPRODUCIBILITY.md', ['Environment Setup', 'Data Preparation']),
        ('CITATION.cff', ['cff-version', 'title', 'authors']),
    ]

    for doc, required_sections in docs_to_check:
        path = root / doc
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            all_present = all(section in content for section in required_sections)
            if check(all_present, f"{doc} has required sections"):
                passed += 1
            else:
                failed += 1
                missing = [s for s in required_sections if s not in content]
                print(f"    Missing: {', '.join(missing)}")
        else:
            failed += 1

    # 5. Check for sensitive data
    print("\n5. Security Checks")

    sensitive_patterns = [
        (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', 'email addresses'),
        (r'[A-Z0-9]{20,}', 'potential API keys'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded passwords'),
    ]

    found_sensitive = False
    for file in root.rglob('*.py'):
        if '__pycache__' in str(file):
            continue

        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        for pattern, name in sensitive_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Filter out common false positives
                real_matches = [m for m in matches if 'example.com' not in m.lower()]
                if real_matches:
                    warn(f"Found {name} in {file.relative_to(root)}: {real_matches[:2]}")
                    found_sensitive = True

    if check(not found_sensitive, "No sensitive data found"):
        passed += 1
    else:
        failed += 1

    # 6. Check imports
    print("\n6. Dependency Checks")

    try:
        import numpy
        import torch
        import scipy
        import sklearn
        import matplotlib
        check(True, "Core dependencies importable")
        passed += 1
    except ImportError as e:
        check(False, f"Dependencies importable: {e}")
        failed += 1

    # Try importing our package
    try:
        import eqgrid
        check(True, "eqgrid package importable")
        passed += 1
    except ImportError as e:
        check(False, f"eqgrid importable: {e}")
        failed += 1

    # 7. Check metadata
    print("\n7. Metadata")

    # Check for placeholders
    readme_path = root / "README.md"
    with open(readme_path, 'r') as f:
        readme = f.read()

    placeholders = ['YOUR_USERNAME', 'XXXXXXX', 'your.email@example.com']
    found_placeholders = [p for p in placeholders if p in readme]

    if found_placeholders:
        warn(f"Found placeholders in README.md: {', '.join(found_placeholders)}")
        warn("Remember to replace before publishing!")

    # 8. Final summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    total = passed + failed
    percentage = (passed / total * 100) if total > 0 else 0

    print(f"\nPassed: {GREEN}{passed}{RESET}")
    print(f"Failed: {RED}{failed}{RESET}")
    print(f"Total: {total}")
    print(f"Success rate: {percentage:.1f}%")

    if failed == 0:
        print(f"\n{GREEN}✓ Package verification passed!{RESET}")
        print("Ready for GitHub/Zenodo release.")
        return 0
    else:
        print(f"\n{RED}✗ Package verification failed.{RESET}")
        print("Please fix the issues above before releasing.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
