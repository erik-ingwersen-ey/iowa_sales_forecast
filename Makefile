.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

# remove all build, test, coverage and Python artifacts
clean: clean-build clean-pyc clean-test

clean-build: # remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: # remove Python file artifacts
	find . -name '*.DS_Store' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.ipynb_checkpoints' -exec rm -fr {} +

clean-test: # remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ##check style with flake8
	flake8 iowa_forecast tests

test: # run tests quickly with the default Python
	pytest

test-all: # run tests on every Python version with tox
	tox

coverage: # check code coverage quickly with the default Python
	coverage run --source iowa_forecast -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: # generate Sphinx HTML documentation, including API docs
	rm -f docs/iowa_forecast.rst
	rm -f docs/modules.rst
	$(sphinx-apidoc) -o docs/iowa_forecast
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/html/index.html

servedocs: docs # compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

install:  # install the package to the active Python's site-packages
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install .
	make set-env

install-dev: # install the package to the active Python's using development mode
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	make set-env

#install:
#	pip install -r ./requirements.txt

update-requirements:
	poetry export --without-hashes --ansi -o ./requirements.txt
