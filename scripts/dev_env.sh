#!/usr/bin/env bash

# Directory dello script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root del progetto
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Rendo disponibile il package Python locale
export PYTHONPATH="${PROJECT_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

# Variabili comode
export SIGMO_PYTHON_ROOT="${PROJECT_ROOT}"
export SIGMO_PYTHON_BUILD_DIR="${PROJECT_ROOT}/build"

echo "Environment ready"
echo "PROJECT_ROOT=${SIGMO_PYTHON_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"