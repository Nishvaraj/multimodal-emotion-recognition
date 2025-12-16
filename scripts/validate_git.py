"""
Git Repository Validation Script
Checks git status and provides initialization guidance
"""
import subprocess
import os
from pathlib import Path

def run_command(command):
    """Run shell command and return output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_git_installed():
    """Check if git is installed"""
    returncode, stdout, stderr = run_command("git --version")
    if returncode == 0:
        print(f"✅ Git installed: {stdout.strip()}")
        return True
    else:
        print("❌ Git not installed")
        print("   Install: https://git-scm.com/downloads")
        return False

def check_git_initialized():
    """Check if git repository is initialized"""
    git_dir = Path(".git")
    if git_dir.exists():
        print("✅ Git repository initialized")
        return True
    else:
        print("❌ Git repository not initialized")
        print("\n   Initialize with:")
        print("   git init")
        print("   git config user.name 'Your Name'")
        print("   git config user.email 'your.email@westminster.ac.uk'")
        return False

def check_git_config():
    """Check git configuration"""
    print("\n" + "=" * 60)
    print("GIT CONFIGURATION")
    print("=" * 60)

    # Check user name
    returncode, username, _ = run_command("git config user.name")
    if returncode == 0 and username.strip():
        print(f"✅ User name: {username.strip()}")
    else:
        print("⚠️  User name not set")
        print("   Set with: git config user.name 'Your Name'")

    # Check user email
    returncode, email, _ = run_command("git config user.email")
    if returncode == 0 and email.strip():
        print(f"✅ User email: {email.strip()}")
    else:
        print("⚠️  User email not set")
        print("   Set with: git config user.email 'your.email@westminster.ac.uk'")

def check_git_status():
    """Check git status"""
    print("\n" + "=" * 60)
    print("GIT STATUS")
    print("=" * 60)

    returncode, stdout, stderr = run_command("git status")
    if returncode == 0:
        print(stdout)

        # Check for untracked/modified files
        if "nothing to commit" in stdout:
            print("✅ Working directory clean")
            return True
        else:
            print("⚠️  You have uncommitted changes")
            print("\n   To commit:")
            print("   git add .")
            print("   git commit -m 'Phase 0: Complete environment setup'")
            return False
    else:
        print(f"❌ Error checking git status: {stderr}")
        return False

def check_commit_history():
    """Check commit history"""
    print("\n" + "=" * 60)
    print("COMMIT HISTORY")
    print("=" * 60)

    returncode, stdout, stderr = run_command("git log --oneline -10")
    if returncode == 0 and stdout.strip():
        lines = stdout.strip().split('\n')
        print(f"✅ {len(lines)} commit(s) found:\n")
        print(stdout)
        return True
    else:
        print("⚠️  No commits yet")
        print("\n   Create initial commit:")
        print("   git add .")
        print("   git commit -m 'Initial commit: Project structure'")
        return False

def check_gitignore():
    """Check if .gitignore exists"""
    print("\n" + "=" * 60)
    print("GITIGNORE CHECK")
    print("=" * 60)

    gitignore = Path(".gitignore")
    if gitignore.exists():
        with open(gitignore, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"✅ .gitignore exists with {len(lines)} rules")

        # Check for critical patterns
        critical_patterns = [
            "__pycache__",
            "*.pyc",
            "venv/",
            ".env",
            "node_modules/",
            "*.log"
        ]

        missing = []
        for pattern in critical_patterns:
            if not any(pattern in line for line in lines):
                missing.append(pattern)

        if missing:
            print(f"\n⚠️  Consider adding these patterns:")
            for pattern in missing:
                print(f"   {pattern}")

        return True
    else:
        print("❌ .gitignore not found")
        print("\n   Create .gitignore to exclude:")
        print("   - Python cache (__pycache__/)")
        print("   - Virtual environment (venv/)")
        print("   - Environment variables (.env)")
        print("   - Node modules (node_modules/)")
        print("   - Large model files (*.pth, *.pt)")
        print("   - Datasets (datasets/)")
        return False

def check_remote():
    """Check for remote repository"""
    print("\n" + "=" * 60)
    print("REMOTE REPOSITORY")
    print("=" * 60)

    returncode, stdout, stderr = run_command("git remote -v")
    if returncode == 0 and stdout.strip():
        print("✅ Remote configured:\n")
        print(stdout)
        return True
    else:
        print("⚠️  No remote repository configured")
        print("\n   Optional: Set up GitHub remote:")
        print("   1. Create repository on GitHub")
        print("   2. git remote add origin <repository-url>")
        print("   3. git push -u origin main")
        return False

def suggest_next_steps(has_commits, is_clean):
    """Suggest next steps based on repository state"""
    print("\n" + "=" * 60)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 60)

    if not has_commits:
        print("1️⃣  Create initial commit:")
        print("    git add .")
        print("    git commit -m 'Phase 0: Initial project setup'")
    elif not is_clean:
        print("1️⃣  Commit current changes:")
        print("    git add .")
        print("    git commit -m 'Phase 0: Environment setup complete'")
    else:
        print("✅ Repository is up to date!")

    print("\n2️⃣  Optional: Push to GitHub for backup")
    print("3️⃣  Proceed to Phase 1: Data Preparation")

def main():
    """Run all git validation checks"""
    print("\n" + "=" * 60)
    print("GIT REPOSITORY VALIDATION")
    print("=" * 60 + "\n")

    # Check git installation
    if not check_git_installed():
        return

    print()

    # Check if initialized
    if not check_git_initialized():
        print("\n❌ Git not initialized. Please run 'git init' first.")
        return

    # Run all checks
    check_git_config()
    has_commits = check_commit_history()
    is_clean = check_git_status()
    check_gitignore()
    check_remote()

    # Suggestions
    suggest_next_steps(has_commits, is_clean)

    print("=" * 60)

if __name__ == "__main__":
    main()
