#!/bin/bash

# format_code.sh - Formatter all Python code in the project using Black,
# excluding Jupyter notebooks and virtual environment directories.
echo "🎨 Formatting Python code with Black..."
echo "========================================="

# Root directory of the project
PROJECT_DIR="/home/lucas/IBelieveICanFlyPy"

# Navigate to the project directory
cd "$PROJECT_DIR" || exit 1

# Format only .py files (exclude .ipynb notebooks)
echo "📂 Formatting files in src/..."
black src/ --exclude '/(\.git|\.venv|venv|__pycache__|\.ipynb_checkpoints)/'

echo ""
echo "📂 Formatting files in setup/..."
black setup/ --exclude '/(\.git|\.venv|venv|__pycache__|\.ipynb_checkpoints)/' 2>/dev/null || echo "⚠️  Directory setup/ not found"

echo ""
echo "📊 Checking formatting..."
black --check src/ setup/ --exclude '/(\.git|\.venv|venv|__pycache__|\.ipynb_checkpoints)/' 2>/dev/null

echo ""
echo "✅ Formatting completed!"
echo "========================================="

# Statistics
TOTAL_FILES=$(find src/ test/ -name "*.py" 2>/dev/null | wc -l)
echo "📊 Total .py files processed: $TOTAL_FILES"