#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Carico l'ambiente locale del progetto
source "${SCRIPT_DIR}/dev_env.sh"

# Configurazione CMake
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}"

# Build
cmake --build "${BUILD_DIR}"

echo "Build completed successfully"