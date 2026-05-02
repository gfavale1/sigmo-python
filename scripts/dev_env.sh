#!/usr/bin/env bash

# Development environment setup for sigmo-python.
#
# This script is meant to be sourced:
#
#   source scripts/dev_env.sh
#
# If executed directly, the environment variables will only be available
# inside the script process and will not persist in the current shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

prepend_path_once() {
    local new_path="$1"
    local current_path="${2:-}"

    case ":${current_path}:" in
        *":${new_path}:"*)
            echo "${current_path}"
            ;;
        *)
            if [[ -n "${current_path}" ]]; then
                echo "${new_path}:${current_path}"
            else
                echo "${new_path}"
            fi
            ;;
    esac
}

export SIGMO_PYTHON_ROOT="${PROJECT_ROOT}"
export SIGMO_PYTHON_BUILD_DIR="${PROJECT_ROOT}/build"
export SIGMO_PYTHON_EXAMPLES_DIR="${PROJECT_ROOT}/examples"
export SIGMO_PYTHON_OUTPUTS_DIR="${PROJECT_ROOT}/examples/outputs"

export PYTHONPATH="$(prepend_path_once "${PROJECT_ROOT}/python" "${PYTHONPATH:-}")"

mkdir -p "${SIGMO_PYTHON_OUTPUTS_DIR}"

if [[ "${SIGMO_QUIET:-0}" != "1" ]]; then
    echo "SIGMo Python development environment ready"
    echo "PROJECT_ROOT=${SIGMO_PYTHON_ROOT}"
    echo "BUILD_DIR=${SIGMO_PYTHON_BUILD_DIR}"
    echo "PYTHONPATH=${PYTHONPATH}"
fi