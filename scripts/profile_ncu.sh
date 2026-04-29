#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark-binary> [benchmark-args...]"
  exit 1
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found in PATH. Please install CUDA Toolkit / Nsight Compute first."
  exit 1
fi

benchmark_bin="$1"
shift

if [[ ! -x "${benchmark_bin}" ]]; then
  echo "Benchmark binary not found or not executable: ${benchmark_bin}"
  exit 1
fi

mkdir -p results/ncu

timestamp="$(date +%Y%m%d_%H%M%S)"
benchmark_name="$(basename "${benchmark_bin}")"
report_base="results/ncu/${benchmark_name}_${timestamp}"

echo "Profiling ${benchmark_name}"
echo "Report base: ${report_base}"

ncu \
  --set full \
  --import-source yes \
  --export "${report_base}" \
  --force-overwrite \
  "${benchmark_bin}" "$@" | tee "${report_base}.stdout.txt"

ncu \
  --import "${report_base}.ncu-rep" \
  --page details | tee "${report_base}.summary.txt"

echo "Saved:"
echo "  ${report_base}.ncu-rep"
echo "  ${report_base}.stdout.txt"
echo "  ${report_base}.summary.txt"
