#!/bin/bash

# Script to run deterministic test with hidden states logging enabled
# Usage: ./run_deterministic_test_with_logging.sh [additional args]

# Set environment variables for hidden states logging
export SGLANG_LOG_HIDDEN_STATES=true
export SGLANG_HIDDEN_STATES_LOG_DIR="./hidden_states_logs"
export SGLANG_MAX_TOKENS_TO_LOG=10
export SGLANG_MAX_DIMS_TO_LOG=20

# Create log directory if it doesn't exist
mkdir -p "$SGLANG_HIDDEN_STATES_LOG_DIR"

echo "=== Hidden States Logging Configuration ==="
echo "SGLANG_LOG_HIDDEN_STATES: $SGLANG_LOG_HIDDEN_STATES"
echo "SGLANG_HIDDEN_STATES_LOG_DIR: $SGLANG_HIDDEN_STATES_LOG_DIR"
echo "SGLANG_MAX_TOKENS_TO_LOG: $SGLANG_MAX_TOKENS_TO_LOG"
echo "SGLANG_MAX_DIMS_TO_LOG: $SGLANG_MAX_DIMS_TO_LOG"
echo "==========================================="

# Run the deterministic test
python3 -m sglang.test.test_deterministic --n-trials 3 --test-mode single "$@"

echo ""
echo "=== Test completed. Check logs in: $SGLANG_HIDDEN_STATES_LOG_DIR ==="
ls -la "$SGLANG_HIDDEN_STATES_LOG_DIR"
