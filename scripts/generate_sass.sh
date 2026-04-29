#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <build-dir> <target-name> [cmake build args...]"
  echo "Example: $0 build/ptx matmul_bench"
  exit 1
fi

build_dir="$1"
target_name="$2"
shift 2

cmake --build "${build_dir}" --target "${target_name}_cubin" "$@"

cubin_path="${build_dir}/cubin/${target_name}.cubin"
sass_dir="${build_dir}/sass"
sass_path="${sass_dir}/${target_name}.sass"

if [[ ! -f "${cubin_path}" ]]; then
  echo "Cubin target finished, but file was not found at ${cubin_path}"
  exit 1
fi

mkdir -p "${sass_dir}"

if command -v nvdisasm >/dev/null 2>&1; then
  nvdisasm -g "${cubin_path}" > "${sass_path}"
elif command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass "${cubin_path}" > "${sass_path}"
else
  echo "Neither nvdisasm nor cuobjdump was found in PATH."
  echo "Please install CUDA Toolkit tools to generate SASS."
  exit 1
fi

echo "Generated cubin: ${cubin_path}"
echo "Generated SASS: ${sass_path}"
