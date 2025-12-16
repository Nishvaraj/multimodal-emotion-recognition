"""
Phase 0 Master Validation Script
Runs all validation checks and provides comprehensive report
"""
import sys
import subprocess
from pathlib import Path

def run_validation_script(script_path, description):
    """Run a validation script and capture results"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout running {script_path}")
        return False
    except Exception as e:
        print(f"❌ Error running {script_path}: {e}")
        return False

def check_directory_structure():
    """Check if required directories exist"""
    print("\n" + "=" * 80)
    print("DIRECTORY STRUCTURE CHECK")
    print("=" * 80)

    required_dirs = [
        "frontend",
        "backend",
        "data",
        "models",
        "notebooks",
        "tests",
        "scripts",
        "logs"
    ]

    all_exist = True
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (missing)")
            all_exist = False

    return all_exist

def check_key_files():
    """Check if key configuration files exist"""
    print("\n" + "=" * 80)
    print("KEY FILES CHECK")
    print("=" * 80)

    key_files = {
        "requirements.txt": "Python dependencies",
        ".gitignore": "Git ignore rules",
        "README.md": "Project documentation",
        "frontend/package.json": "Frontend dependencies"
    }

    all_exist = True
    for file_path, description in key_files.items():
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({description}) - {size} bytes")
        else:
            print(f"⚠️  {file_path} ({description}) - MISSING")
            if file_path == "README.md":
                all_exist = False

    return all_exist

def generate_phase0_completion_report():
    """Generate Phase 0 completion report"""
    print("\n" + "=" * 80)
    print("PHASE 0 COMPLETION REPORT")
    print("=" * 80)

    checklist = {
        "Environment Setup": [
            ("Python 3.10+ installed", "Run: python --version"),
            ("Virtual environment created", "Check: venv/ directory exists"),
            ("Dependencies installed", "Validated by validate_setup.py")
        ],
        "Project Structure": [
            ("Directory structure created", "All folders present"),
            ("Git repository initialized", "Validated by validate_git.py"),
            (".gitignore configured", ".gitignore file exists")
        ],
        "Datasets": [
            ("FER2013 downloaded", "Validated by validate_datasets.py"),
            ("RAVDESS downloaded", "Validated by validate_datasets.py"),
            ("Data structure correct", "Emotion folders organized")
        ],
        "Documentation": [
            ("README.md created", "README.md exists"),
            ("Project info documented", "Quick start, structure, milestones"),
            ("Git commits made", "Initial commit present")
        ]
    }

    for category, items in checklist.items():
        print(f"\n📋 {category}:")
        for item, verification in items:
            print(f"   □ {item}")
            print(f"      → {verification}")

def main():
    """Run complete Phase 0 validation"""
    print("\n" + "🎯" + "=" * 78 + "🎯")
    print("PHASE 0: PRE-IMPLEMENTATION VALIDATION")
    print("Multi-Modal Emotion Recognition Project")
    print("🎯" + "=" * 78 + "🎯")

    results = {}

    # 1. Check directory structure
    results['directories'] = check_directory_structure()

    # 2. Check key files
    results['files'] = check_key_files()

    # 3. Run validation scripts
    scripts_dir = Path("scripts")

    validation_scripts = [
        ("validate_setup.py", "Dependencies Validation"),
        ("validate_datasets.py", "Datasets Validation"),
        ("validate_git.py", "Git Repository Validation")
    ]

    for script_name, description in validation_scripts:
        script_path = scripts_dir / script_name
        if script_path.exists():
            results[script_name] = run_validation_script(str(script_path), description)
        else:
            print(f"\n⚠️  {script_path} not found - skipping")
            results[script_name] = False

    # Generate completion report
    generate_phase0_completion_report()

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    summary = [
        ("Directory Structure", results.get('directories', False)),
        ("Key Files", results.get('files', False)),
        ("Dependencies", results.get('validate_setup.py', False)),
        ("Datasets", results.get('validate_datasets.py', False)),
        ("Git Repository", results.get('validate_git.py', False))
    ]

    passed_count = sum(1 for _, status in summary if status)
    total_count = len(summary)

    for check_name, status in summary:
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{check_name:25s}: {status_str}")

    print(f"\nScore: {passed_count}/{total_count} checks passed")

    # Final verdict
    print("\n" + "=" * 80)
    if passed_count >= 4:  # Allow README to be pending
        print("🎉 PHASE 0 VALIDATION PASSED!")
        print("✅ Ready to proceed to Phase 1: Data Preparation & Baseline")
        print("\nNext Steps:")
        print("  1. Create/update README.md if not done")
        print("  2. Commit any remaining changes")
        print("  3. Start Phase 1: Create notebooks/01_data_exploration.ipynb")
    else:
        print("⚠️  PHASE 0 INCOMPLETE")
        print("Please address the failed checks above before proceeding.")
        print("\nCritical items to fix:")
        for check_name, status in summary:
            if not status:
                print(f"  • {check_name}")

    print("=" * 80)

if __name__ == "__main__":
    main()
