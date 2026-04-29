#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage: inspect_kernel.sh <build-dir> <target-name> [options] [-- benchmark-args...]

Options:
  --configure                  Run cmake configure before build
  --configure-preset <name>    Configure with a CMake preset
  --build-preset <name>        Build with a CMake preset instead of build-dir
  --skip-run                   Skip running the benchmark binary
  --skip-ptx                   Skip PTX generation
  --skip-sass                  Skip SASS generation
  --skip-ncu                   Skip Nsight Compute profiling
  --ncu-build-dir <dir>        Build directory to use for ncu artifacts, default is <build-dir>

Examples:
  inspect_kernel.sh build/profile matmul_bench -- --m 2048 --n 2048 --k 2048 --tile 16
  inspect_kernel.sh build/ptx matmul_bench --skip-ncu -- --m 1024 --n 1024 --k 1024
  inspect_kernel.sh build/profile matmul_bench --configure -- --m 2048 --n 2048 --k 2048
EOF
  exit 1
fi

build_dir="$1"
target_name="$2"
shift 2

configure_requested=0
configure_preset=""
build_preset=""
skip_run=0
skip_ptx=0
skip_sass=0
skip_ncu=0
ncu_build_dir=""
benchmark_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure)
      configure_requested=1
      shift
      ;;
    --configure-preset)
      configure_preset="${2:-}"
      shift 2
      ;;
    --build-preset)
      build_preset="${2:-}"
      shift 2
      ;;
    --skip-run)
      skip_run=1
      shift
      ;;
    --skip-ptx)
      skip_ptx=1
      shift
      ;;
    --skip-sass)
      skip_sass=1
      shift
      ;;
    --skip-ncu)
      skip_ncu=1
      shift
      ;;
    --ncu-build-dir)
      ncu_build_dir="${2:-}"
      shift 2
      ;;
    --)
      shift
      benchmark_args=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${ncu_build_dir}" ]]; then
  ncu_build_dir="${build_dir}"
fi

run_build() {
  if [[ -n "${build_preset}" ]]; then
    cmake --build --preset "${build_preset}"
  else
    cmake --build "${build_dir}" --target "${target_name}"
  fi
}

if [[ ${configure_requested} -eq 1 ]]; then
  if [[ -n "${configure_preset}" ]]; then
    cmake --preset "${configure_preset}"
  else
    cmake -S . -B "${build_dir}"
  fi
fi

echo "==> Building ${target_name}"
run_build

binary_path="${build_dir}/bin/${target_name}"
if [[ ! -x "${binary_path}" ]]; then
  echo "Benchmark binary not found or not executable: ${binary_path}"
  exit 1
fi

if [[ ${skip_run} -eq 0 ]]; then
  echo "==> Running ${target_name}"
  "${binary_path}" "${benchmark_args[@]}"
fi

if [[ ${skip_ptx} -eq 0 ]]; then
  echo "==> Generating PTX"
  "$(dirname "$0")/generate_ptx.sh" "${build_dir}" "${target_name}"
fi

if [[ ${skip_sass} -eq 0 ]]; then
  echo "==> Generating SASS"
  "$(dirname "$0")/generate_sass.sh" "${build_dir}" "${target_name}"
fi

if [[ ${skip_ncu} -eq 0 ]]; then
  ncu_binary_path="${ncu_build_dir}/bin/${target_name}"
  if [[ ! -x "${ncu_binary_path}" ]]; then
    echo "Skipping ncu because binary was not found at ${ncu_binary_path}"
  elif ! command -v ncu >/dev/null 2>&1; then
    echo "Skipping ncu because ncu is not available in PATH"
  else
    echo "==> Profiling with Nsight Compute"
    "$(dirname "$0")/profile_ncu.sh" "${ncu_binary_path}" "${benchmark_args[@]}"
  fi
fi
