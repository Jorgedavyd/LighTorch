#!/bin/bash

# Remove existing requirements.txt if it exists
rm -f requirements.txt

# Generate requirements.in from the lightorch directory
pipreqs lightorch/ --savepath=requirements.in

# Compile requirements.in to requirements.txt
pip-compile requirements.in

# Clean up
rm -f requirements.in