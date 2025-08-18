#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Setting up development environment..."

# 1. Install golangci-lint if not present
if ! command -v golangci-lint &> /dev/null; then
    echo "Installing golangci-lint..."
    # Install the latest version
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
    echo -e "${GREEN}✓${NC} golangci-lint installed"
else
    echo -e "${GREEN}✓${NC} golangci-lint already installed ($(golangci-lint version))"
fi

# 2. Install gofumpt if not present
if ! command -v gofumpt &> /dev/null; then
    echo "Installing gofumpt..."
    go install mvdan.cc/gofumpt@latest
    echo -e "${GREEN}✓${NC} gofumpt installed"
else
    echo -e "${GREEN}✓${NC} gofumpt already installed"
fi

# 3. Install git hooks
echo "Installing git hooks..."

# Create .git/hooks directory if it doesn't exist
mkdir -p "$REPO_ROOT/.git/hooks"

# Symlink pre-commit hook
if [ -L "$REPO_ROOT/.git/hooks/pre-commit" ] || [ -f "$REPO_ROOT/.git/hooks/pre-commit" ]; then
    echo "Removing existing pre-commit hook..."
    rm "$REPO_ROOT/.git/hooks/pre-commit"
fi

ln -s "$REPO_ROOT/scripts/pre-commit" "$REPO_ROOT/.git/hooks/pre-commit"
echo -e "${GREEN}✓${NC} Pre-commit hook installed"

echo ""
echo -e "${YELLOW}Setup complete!${NC}"
echo ""
echo "The pre-commit hook will run automatically on 'git commit'"
echo "To bypass the pre-commit hook (not recommended), use:"
echo "  git commit --no-verify"