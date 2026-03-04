#!/usr/bin/env python3
"""Fix all imports in sam2 module from 'models.sam2' to 'sam_tss.models.sam2'"""

import re
from pathlib import Path


def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace all variations
    content = re.sub(r'from models\.sam2\.', 'from sam_tss.models.sam2.', content)
    content = re.sub(r'import models\.sam2 as sam2', 'import sam_tss.models.sam2 as sam2', content)
    content = re.sub(r'^import models\.sam2$', 'import sam_tss.models.sam2', content, flags=re.MULTILINE)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    sam2_dir = Path(__file__).parent / "src" / "sam_tss" / "models" / "sam2"
    
    if not sam2_dir.exists():
        print(f"Error: Directory not found: {sam2_dir}")
        return
    
    print(f"Fixing imports in: {sam2_dir}")
    print("=" * 70)
    
    fixed_count = 0
    total_count = 0
    
    for py_file in sam2_dir.rglob("*.py"):
        total_count += 1
        rel_path = py_file.relative_to(sam2_dir)
        if fix_imports_in_file(py_file):
            fixed_count += 1
            print(f"✓ Fixed: {rel_path}")
    
    print("=" * 70)
    print(f"Fixed {fixed_count}/{total_count} files")
    print("\nNow try: uv run instantiate_rtmvss.py")


if __name__ == "__main__":
    main()
