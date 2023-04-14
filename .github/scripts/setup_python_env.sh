#!/bin/bash

python -m pip install --upgrade pip
pip install -e ".[pytorch, huggingface]"
pip install -r requirements-dev.txt
