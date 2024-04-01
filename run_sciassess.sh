#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PYTHONPATH="${SCRIPT_DIR}"
MODEL="${1:-gpt3.5-withPDF}"

scievalset "$MODEL" alloy_materials --registry_path "${SCRIPT_DIR}/sciassess/Registry"
scievalset "$MODEL" biomedicine --registry_path "${SCRIPT_DIR}/sciassess/Registry"
scievalset "$MODEL",gpt-4-1106-preview drug_discovery --registry_path "${SCRIPT_DIR}/sciassess/Registry"
scievalset "$MODEL",gpt-4-1106-preview general_chemistry --registry_path "${SCRIPT_DIR}/sciassess/Registry"
scievalset "$MODEL" organic_materials --registry_path "${SCRIPT_DIR}/sciassess/Registry"
