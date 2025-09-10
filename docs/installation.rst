Installation Guide
==================

This guide will help you get the Stock Analysis App up and running on your system.

Prerequisites
-------------

Before installing the Stock Analysis App, ensure you have the following:

- **Python 3.9 or higher**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

.. code-block:: bash

   # Check Python version
   python --version

   # Check pip version
   pip --version

Installation Methods
--------------------

Method 1: Direct Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/JustR3/stock_analysis_app.git
      cd stock_analysis_app

2. **Create a virtual environment:**

   .. code-block:: bash

      # Using venv (recommended)
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

      # Or using conda
      conda create -n stock-analysis python=3.9
      conda activate stock-analysis

3. **Install the package:**

   .. code-block:: bash

      # Install in development mode (recommended for contributors)
      pip install -e .

      # Or install with all optional dependencies
      pip install -e .[dev,test,docs,ml]

Method 2: Using Make (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have Make installed, you can use the provided Makefile for easier installation:

.. code-block:: bash

   # Clone and setup everything
   git clone https://github.com/JustR3/stock_analysis_app.git
   cd stock_analysis_app

   # Setup development environment (includes virtual environment)
   make setup-dev

   # Or install all dependencies
   make install-all

Method 3: Using Docker
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build Docker image
   make docker-build

   # Run the application
   make docker-run

Verification
------------

After installation, verify that everything is working:

.. code-block:: bash

   # Check if the CLI tool is available
   stock-analysis --help

   # Run a quick test
   python -c "import stock_analysis_app; print('Installation successful!')"

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**

If you encounter import errors, make sure you're in the correct directory and have activated your virtual environment:

.. code-block:: bash

   cd /path/to/stock_analysis_app
   source venv/bin/activate
   pip install -e .

**Missing Dependencies**

If some packages fail to install, try updating pip and setuptools:

.. code-block:: bash

   pip install --upgrade pip setuptools wheel

**Permission Errors**

On some systems, you might need to install packages with user permissions:

.. code-block:: bash

   pip install --user -e .

Next Steps
----------

After successful installation:

1. **Run the web application:**

   .. code-block:: bash

      make app

2. **Explore the documentation:**

   .. code-block:: bash

      make docs-serve

3. **Run the tests:**

   .. code-block:: bash

      make test

For more information, see the :doc:`quickstart` guide.
