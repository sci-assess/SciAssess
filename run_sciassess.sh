#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PYTHONPATH="${SCRIPT_DIR}"
export USE_COT=1
MODEL="${1:-gpt3.5}"

scievalset "$MODEL" biology --registry_path "${SCRIPT_DIR}/sciassess/Registry"
scievalset "$MODEL" chemistry --registry_path "${SCRIPT_DIR}/sciassess/Registry"
scievalset "$MODEL" material --registry_path "${SCRIPT_DIR}/sciassess/Registry"
scievalset "$MODEL" medicine --registry_path "${SCRIPT_DIR}/sciassess/Registry"
python collect "$MODEL"
