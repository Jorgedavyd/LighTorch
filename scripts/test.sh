#!/bin/bash
pytest tests/
find . -type d -name "__pycache__" -exec rm -rf {} +

