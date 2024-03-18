#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PYTHONPATH="${SCRIPT_DIR}"
MODEL="${1:-gpt-3.5-withPDF}"

oaievalset "$MODEL" alloy_materials --registry_path "${SCRIPT_DIR}/sciassess/Registry"
oaievalset "$MODEL" biomedicine --registry_path "${SCRIPT_DIR}/sciassess/Registry"
oaievalset "$MODEL" drug_discovery --registry_path "${SCRIPT_DIR}/sciassess/Registry"
oaievalset "$MODEL" general_chemistry --registry_path "${SCRIPT_DIR}/sciassess/Registry"
oaievalset "$MODEL" organic_materials --registry_path "${SCRIPT_DIR}/sciassess/Registry"
