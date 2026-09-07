"""Automated dependency verification and smoke test script."""
import subprocess
import sys


def check_pip_conflicts():
    """Run pip check to ensure no dependency graph conflicts."""
    print("🔍 Running pip check...")
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if res.returncode == 0:
        print("✅ pip check: No broken requirements or version conflicts found.\n")
        return True
    else:
        print("❌ pip check: Conflicts detected!")
        print(res.stdout)
        print(res.stderr)
        return False


def test_core_library_imports():
    """Smoke test critical AI/ML libraries to ensure binary compatibility."""
    critical_libraries = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("psycopg2", "PostgreSQL Adapter"),
        ("redis", "Redis Client"),
    ]

    print("🧪 Running library smoke tests...")
    failed = []
    for module_name, display_name in critical_libraries:
        try:
            __import__(module_name)
            print(f"   ✓ {display_name:<25} [OK]")
        except Exception as e:
            print(f"   ❌ {display_name:<25} [FAILED: {e}]")
            failed.append(display_name)

    if not failed:
        print("\n✅ All critical library imports passed!\n")
        return True
    else:
        print(f"\n❌ {len(failed)} library import(s) failed: {', '.join(failed)}\n")
        return False


def main():
    pip_ok = check_pip_conflicts()
    imports_ok = test_core_library_imports()

    if pip_ok and imports_ok:
        print("🎉 Dependency validation 100% SUCCESSFUL!")
        sys.exit(0)
    else:
        print("⚠️ Dependency validation FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
