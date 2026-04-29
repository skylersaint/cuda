#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <build-dir> <target-name> [cmake build args...]"
  echo "Example: $0 build/profile matmul_bench"
  exit 1
fi

build_dir="$1"
target_name="$2"
shift 2

cmake --build "${build_dir}" --target "${target_name}_ptx" "$@"

ptx_path="${build_dir}/ptx/${target_name}.ptx"
if [[ -f "${ptx_path}" ]]; then
  echo "Generated PTX: ${ptx_path}"
else
  echo "PTX target finished, but file was not found at ${ptx_path}"
  exit 1
fi
