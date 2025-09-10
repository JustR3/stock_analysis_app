.PHONY: help install install-dev install-test clean lint format test test-cov coverage docs build publish app pre-commit

# Default target
help: ## Show this help message
	@echo "Stock Analysis App - Development Commands"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Installation
install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e .[dev]

install-test: ## Install test dependencies
	pip install -e .[test]

install-docs: ## Install documentation dependencies
	pip install -e .[docs]

install-all: ## Install all dependencies
	pip install -e .[dev,test,docs,ml]

# Code quality
lint: ## Run linting (flake8)
	flake8 src/ tests/ --max-line-length=88

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking with mypy
	mypy src/stock_analysis_app/

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# Testing
test: ## Run tests with pytest
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src/stock_analysis_app --cov-report=term-missing --cov-report=html

test-unit: ## Run only unit tests
	pytest tests/unit/ -v

test-integration: ## Run only integration tests
	pytest tests/integration/ -v

# Coverage
coverage: ## Generate coverage report
	coverage run -m pytest tests/
	coverage report
	coverage html

# Documentation
docs: ## Build documentation
	sphinx-build -b html docs/source docs/build/html

docs-serve: ## Serve documentation locally
	cd docs/build/html && python -m http.server 8000

# Building and publishing
build: ## Build the package
	python -m build

publish-test: ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	python -m twine upload dist/*

# Application
app: ## Run the Streamlit app
	streamlit run src/streamlit_app/Home.py

app-prod: ## Run the app in production mode
	streamlit run src/streamlit_app/Home.py --server.port 8501 --server.address 0.0.0.0

# Development workflow
setup-dev: clean install-dev pre-commit ## Set up development environment
	@echo "Development environment setup complete!"

setup-pre-commit: ## Set up pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# Data operations
download-data: ## Download sample data
	python scripts/data/download_sample_data.py

clean-data: ## Clean cached data
	rm -rf data/cache/*
	rm -rf data/raw/*

# Model operations
train-models: ## Train all models
	python scripts/ml/train_models.py

evaluate-models: ## Evaluate trained models
	python scripts/ml/evaluate_models.py

# Utility
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	rm -rf src/**/*.pyc
	rm -rf src/**/__pycache__/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

clean-all: clean ## Clean everything including venv and data
	rm -rf venv/
	rm -rf data/cache/
	rm -rf data/raw/
	rm -rf models/*.pkl
	rm -rf models/*.joblib

# Environment
venv: ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate' to activate."

activate: ## Show how to activate virtual environment
	@echo "Run: source venv/bin/activate"

# Docker
docker-build: ## Build Docker image
	docker build -t stock-analysis-app .

docker-run: ## Run Docker container
	docker run -p 8501:8501 stock-analysis-app

# CI/CD simulation
ci: lint type-check test-cov build ## Run full CI pipeline locally

# Help for specific commands
help-%: ## Show help for a specific command
	@grep -E '^$*:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^$*:' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
