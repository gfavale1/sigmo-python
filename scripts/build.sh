#!/usr/bin/env bash
set -euo pipefail

# Build script for sigmo-python.
#
# Usage:
#
#   ./scripts/build.sh
#
# Optional environment variables:
#
#   BUILD_DIR=build-debug ./scripts/build.sh
#   BUILD_TYPE=Debug ./scripts/build.sh
#   JOBS=4 ./scripts/build.sh
#
# Optional CMake arguments:
#
#   ./scripts/build.sh -DCMAKE_CXX_COMPILER=icpx

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIGMO_QUIET=1 source "${SCRIPT_DIR}/dev_env.sh"

BUILD_DIR="${BUILD_DIR:-${SIGMO_PYTHON_BUILD_DIR}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake not found in PATH." >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/CMakeLists.txt" ]]; then
    echo "Error: CMakeLists.txt not found in ${PROJECT_ROOT}." >&2
    exit 1
fi

echo "Building SIGMo Python project"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "JOBS=${JOBS}"

echo
echo "[1/2] Configuring CMake"
cmake \
    -S "${PROJECT_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    "$@"

echo
echo "[2/2] Building"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo
echo "Build completed successfully"
echo
echo "Install the Python package in editable mode:"
echo "  python -m pip install -e . --no-build-isolation --no-deps"
echo
echo "Then verify with:"
echo "  python -c \"import sigmo; print('SIGMo import OK')\""