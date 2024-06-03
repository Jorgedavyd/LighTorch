#!/bin/bash

# Set the project directory and requirements file paths
PROJECT_DIR="lightorch"
REQUIREMENTS_IN="requirements.in"
REQUIREMENTS_TXT="requirements.txt"

# Check if pipreqs is installed
if ! command -v pipreqs &> /dev/null; then
    echo "pipreqs could not be found. Please install it with 'pip install pipreqs'."
    exit 1
fi

# Generate requirements.in using pipreqs
echo "Generating ${REQUIREMENTS_IN} with pipreqs..."
pipreqs "$PROJECT_DIR" --force --savepath="$REQUIREMENTS_IN"

# Check if pipreqs succeeded
if [[ $? -ne 0 ]]; then
    echo "pipreqs failed to generate ${REQUIREMENTS_IN}. Exiting."
    exit 1
fi

# Check if requirements.in exists and is not empty
if [[ ! -s "$REQUIREMENTS_IN" ]]; then
    echo "${REQUIREMENTS_IN} is empty or does not exist. Exiting."
    exit 1
fi

# Compile requirements.in to requirements.txt using pip-compile
echo "Compiling ${REQUIREMENTS_IN} to ${REQUIREMENTS_TXT} with pip-compile..."
pip-compile --output-file="$REQUIREMENTS_TXT" "$REQUIREMENTS_IN"

# Check if pip-compile succeeded
if [[ $? -ne 0 ]]; then
    echo "pip-compile failed to generate ${REQUIREMENTS_TXT}. Exiting."
    exit 1
fi

# Remove the intermediate requirements.in file
echo "Removing ${REQUIREMENTS_IN}..."
rm -f "$REQUIREMENTS_IN"

echo "Requirements successfully generated in ${REQUIREMENTS_TXT}."
