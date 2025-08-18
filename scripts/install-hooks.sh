#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Setting up development environment..."

# Determine where go install puts binaries
GOPATH=$(go env GOPATH)
GOBIN=$(go env GOBIN)
if [ -z "$GOBIN" ]; then
    INSTALL_DIR="$GOPATH/bin"
else
    INSTALL_DIR="$GOBIN"
fi

# Check if install directory is in PATH
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    echo -e "${YELLOW}Warning: $INSTALL_DIR is not in your PATH${NC}"
    echo "Add this to your shell profile (.bashrc, .zshrc, etc.):"
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    echo ""
fi

# 1. Install staticcheck if not present
if ! command -v staticcheck &> /dev/null; then
    echo "Installing staticcheck to $INSTALL_DIR..."
    go install honnef.co/go/tools/cmd/staticcheck@latest
    
    # Check if it's now available
    if [ -f "$INSTALL_DIR/staticcheck" ]; then
        echo -e "${GREEN}✓${NC} staticcheck installed to $INSTALL_DIR"
        if ! command -v staticcheck &> /dev/null; then
            echo -e "${YELLOW}   Note: You may need to add $INSTALL_DIR to your PATH or restart your shell${NC}"
        fi
    else
        echo -e "${RED}Failed to install staticcheck${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} staticcheck already installed"
fi

# 2. Install gofumpt if not present
if ! command -v gofumpt &> /dev/null; then
    echo "Installing gofumpt to $INSTALL_DIR..."
    go install mvdan.cc/gofumpt@latest
    
    # Check if it's now available
    if [ -f "$INSTALL_DIR/gofumpt" ]; then
        echo -e "${GREEN}✓${NC} gofumpt installed to $INSTALL_DIR"
        if ! command -v gofumpt &> /dev/null; then
            echo -e "${YELLOW}   Note: You may need to add $INSTALL_DIR to your PATH or restart your shell${NC}"
        fi
    else
        echo -e "${RED}Failed to install gofumpt${NC}"
        exit 1
    fi
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
