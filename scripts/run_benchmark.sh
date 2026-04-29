#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark-binary> [args...]"
  exit 1
fi

benchmark_bin="$1"
shift

if [[ ! -x "${benchmark_bin}" ]]; then
  echo "Benchmark binary not found or not executable: ${benchmark_bin}"
  exit 1
fi

"${benchmark_bin}" "$@"

