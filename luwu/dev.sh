#!/bin/bash

# Development script for luwu project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup development environment
setup() {
    print_status "Setting up development environment..."
    
    # Check if PDM is installed
    if ! command_exists pdm; then
        print_error "PDM is not installed. Please install it first:"
        echo "curl -sSL https://pdm-project.org/install-pdm.py | python3 -"
        exit 1
    fi
    
    # Install dependencies
    print_status "Installing dependencies..."
    pdm install
    
    # Install development dependencies
    print_status "Installing development dependencies..."
    pdm install -G dev
    
    # Setup pre-commit hooks
    if command_exists pre-commit; then
        print_status "Setting up pre-commit hooks..."
        pdm run pre-commit install
    else
        print_warning "pre-commit not found. Install it for better development experience."
    fi
    
    print_success "Development environment setup completed!"
}

# Format code
format() {
    print_status "Formatting code..."
    
    print_status "Running black..."
    pdm run black src tests examples || true
    
    print_status "Running isort..."
    pdm run isort src tests examples || true
    
    print_success "Code formatting completed!"
}

# Lint code
lint() {
    print_status "Linting code..."
    
    print_status "Running ruff..."
    pdm run ruff check src tests examples || true
    
    print_status "Running mypy..."
    pdm run mypy src || true
    
    print_success "Code linting completed!"
}

# Run tests
test() {
    print_status "Running tests..."
    
    # Run tests with coverage
    pdm run pytest tests/ --cov=luwu --cov-report=term-missing --cov-report=html
    
    print_success "Tests completed!"
}

# Run all quality checks
check() {
    print_status "Running all quality checks..."
    
    format
    lint
    test
    
    print_success "All quality checks completed!"
}

# Clean up generated files
clean() {
    print_status "Cleaning up generated files..."
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove test artifacts
    rm -rf .coverage htmlcov/ .pytest_cache/
    
    # Remove build artifacts
    rm -rf build/ dist/ *.egg-info/
    
    print_success "Cleanup completed!"
}

# Build package
build() {
    print_status "Building package..."
    
    # Clean first
    clean
    
    # Build
    pdm build
    
    print_success "Package built successfully!"
}

# Install package in development mode
install() {
    print_status "Installing package in development mode..."
    
    pdm install -e .
    
    print_success "Package installed successfully!"
}

# Run training example
train_example() {
    print_status "Running training example..."
    
    # Make sure package is installed
    pdm install -e .
    
    # Run example
    pdm run python examples/train_example.py
    
    print_success "Training example completed!"
}

# Show usage
usage() {
    echo "Usage: $0 {setup|format|lint|test|check|clean|build|install|train-example|help}"
    echo ""
    echo "Commands:"
    echo "  setup        - Set up development environment"
    echo "  format       - Format code with black and isort"
    echo "  lint         - Lint code with ruff and mypy"
    echo "  test         - Run tests with coverage"
    echo "  check        - Run all quality checks (format, lint, test)"
    echo "  clean        - Clean up generated files"
    echo "  build        - Build package"
    echo "  install      - Install package in development mode"
    echo "  train-example - Run training example"
    echo "  help         - Show this help message"
}

# Main script logic
case "${1:-help}" in
    setup)
        setup
        ;;
    format)
        format
        ;;
    lint)
        lint
        ;;
    test)
        test
        ;;
    check)
        check
        ;;
    clean)
        clean
        ;;
    build)
        build
        ;;
    install)
        install
        ;;
    train-example)
        train_example
        ;;
    help)
        usage
        ;;
    *)
        print_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
