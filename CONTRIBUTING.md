# Contributing to Stock Analysis App

Thank you for your interest in contributing to the Stock Analysis App! We welcome contributions from everyone. This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## 🤝 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Make (optional, but recommended)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/stock_analysis_app.git
   cd stock_analysis_app
   ```

3. Add the upstream remote:

   ```bash
   git remote add upstream https://github.com/JustR3/stock_analysis_app.git
   ```

## 🛠️ Development Setup

### Automated Setup

```bash
# Setup development environment
make setup-dev

# This will:
# - Create virtual environment
# - Install all dependencies
# - Setup pre-commit hooks
# - Run initial tests
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev,test,docs,ml]

# Setup pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run tests
pytest tests/
```

## 🔄 Making Changes

### Branch Naming

Use descriptive branch names following this pattern:

```bash
git checkout -b feature/add-new-indicator
git checkout -b bugfix/fix-rsi-calculation
git checkout -b docs/update-api-docs
git checkout -b refactor/cleanup-imports
```

### Code Style

This project uses automated code formatting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting

```bash
# Format code
make format

# Check code quality
make lint
```

### Commit Messages

Follow conventional commit format:

```bash
# Good examples
feat: add new technical indicator
fix: correct RSI calculation formula
docs: update API documentation
refactor: clean up data processing logic

# Bad examples
fixed bug
updated code
changes
```

## 📝 Submitting Changes

### Pull Request Process

1. **Update your branch**:

   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run tests and checks**:

   ```bash
   make ci  # Runs full CI pipeline locally
   ```

3. **Push your changes**:

   ```bash
   git push origin your-branch
   ```

4. **Create Pull Request**:

   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Fill out the PR template
   - Request review from maintainers

### PR Requirements

- [ ] All CI checks pass
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated if needed
- [ ] Code follows style guidelines
- [ ] Commit messages are clear and descriptive
- [ ] PR description follows template

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
make test-unit
make test-integration
```

### Writing Tests

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Use descriptive test names
- Follow `test_function_name` naming convention
- Use pytest fixtures for setup/teardown

### Test Coverage

Aim for high test coverage. Current requirements:
- Minimum 80% overall coverage
- All new code must be tested
- Critical paths should have 90%+ coverage

## 📚 Documentation

### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

### Documentation Guidelines

- Use Google-style docstrings
- Include type hints
- Document parameters and return values
- Add examples for complex functions
- Keep documentation up-to-date

## 🐛 Reporting Issues

### Bug Reports

Use the bug report template and include:

- Clear title describing the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Screenshots if applicable
- Error messages/logs

### Feature Requests

Use the feature request template and include:

- Clear description of the proposed feature
- Use case and benefits
- Implementation suggestions
- Acceptance criteria

## 🎯 Development Workflow

### Daily Development

```bash
# Start your day
git checkout main
git pull upstream main

# Work on feature
git checkout -b feature/your-feature
# Make changes...

# Before committing
make format
make lint
make test

# Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature
```

### Code Review Process

1. **Author** creates PR with description
2. **Reviewers** provide feedback
3. **Author** addresses feedback
4. **Maintainers** approve and merge
5. **PR** is automatically closed

## 🏆 Recognition

Contributors are recognized through:
- GitHub contributor statistics
- Mention in release notes
- Attribution in documentation
- Community recognition

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: For API and usage information
- **Email**: team@stockanalysis.com for private matters

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

Thank you for contributing to the Stock Analysis App! 🚀
