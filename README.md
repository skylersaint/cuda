# CUDA Benchmark Baseline

一个适合持续添加 CUDA kernel、做 benchmark、再接 Nsight Compute 分析的仓库骨架。

## 目录结构

```text
.
├── benchmarks/
│   ├── benchmark_template.cu
│   ├── matmul_bench.cu
│   └── vector_add_bench.cu
├── CMakeLists.txt
├── CMakePresets.json
├── Dockerfile
├── include/
│   ├── cuda_benchmark.cuh
│   └── cuda_utils.cuh
├── scripts/
│   ├── inspect_kernel.sh
│   ├── generate_ptx.sh
│   ├── generate_sass.sh
│   ├── profile_ncu.sh
│   └── run_benchmark.sh
└── README.md
```

## 设计目标

- 方便你持续新增 kernel benchmark
- 默认带 warmup / 多次迭代 / 统计结果
- 直接兼容 Nsight Compute (`ncu`) 进行 profile
- 支持在不同 NVIDIA GPU 上按架构编译

## 本地编译

前提：

- 安装 NVIDIA Driver
- 安装 CUDA Toolkit（需要 `nvcc` / `ncu`）
- 安装 CMake 3.24+
- 安装 Ninja（`CMakePresets.json` 默认使用 Ninja）

```bash
cmake --preset release
cmake --build --preset release
./build/release/bin/vector_add_bench
```

也可以直接跑 `matmul` baseline：

```bash
./build/release/bin/matmul_bench --m 1024 --n 1024 --k 1024 --tile 16 --warmup 10 --iters 50
```

如果你要为 Nsight Compute 准备一个更友好的构建目录：

```bash
cmake --preset profile
cmake --build --preset profile
./build/profile/bin/vector_add_bench
```

如果你也希望后面直接查看 PTX，推荐在 configure 时显式指定架构：

```bash
cmake -S . -B build/ptx -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_PTX_ARCHITECTURE=80
cmake --build build/ptx
```

## 常用构建方式

默认会打开 `-lineinfo`，方便 Nsight Compute 把指标映射回源码。

```bash
cmake --preset release
cmake --build --preset release
```

如果你想显式指定 GPU 架构：

```bash
cmake -S . -B build/sm80 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/sm80
./build/sm80/bin/vector_add_bench
```

常见示例：

- `75` -> T4 / RTX 20 系列
- `80` -> A100
- `86` -> RTX 30 系列
- `89` -> RTX 40 部分型号
- `90` -> H100

## 运行 benchmark

```bash
./build/release/bin/vector_add_bench --elements 16777216 --threads 256 --warmup 20 --iters 100
```

支持参数：

- `--elements <N>`: 输入元素数
- `--threads <N>`: 每个 block 的线程数
- `--warmup <N>`: warmup 次数
- `--iters <N>`: 正式计时次数
- `--no-check`: 跳过结果校验

`matmul_bench` 支持参数：

- `--m <M>`: 输出矩阵行数
- `--n <N>`: 输出矩阵列数
- `--k <K>`: 归约维度
- `--tile <T>`: tile 大小，同时也是 `blockDim.x == blockDim.y`
- `--warmup <N>`: warmup 次数
- `--iters <N>`: 正式计时次数
- `--no-check`: 跳过结果校验

## 生成 PTX

每个 benchmark 都会自动带一个同名的 PTX target：

- `vector_add_bench_ptx`
- `matmul_bench_ptx`

例如生成 `matmul` 的 PTX：

```bash
cmake -S . -B build/ptx -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_PTX_ARCHITECTURE=80
./scripts/generate_ptx.sh build/ptx matmul_bench
```

生成结果默认在：

```text
build/ptx/ptx/matmul_bench.ptx
```

你也可以直接用 CMake target：

```bash
cmake --build build/ptx --target matmul_bench_ptx
cmake --build build/ptx --target vector_add_bench_ptx
```

建议：

- 看 PTX 时尽量显式指定 `CUDA_PTX_ARCHITECTURE`
- 和 `-lineinfo` 一起用，方便把 PTX、源码和 `ncu` 结果对上
- 如果后面你要看更底层的机器码，再补 `cuobjdump` / `nvdisasm` 工作流

## 生成 SASS

每个 benchmark 也会自动带一个同名的 cubin target：

- `vector_add_bench_cubin`
- `matmul_bench_cubin`

例如生成 `matmul` 的 cubin 和 SASS：

```bash
cmake -S . -B build/ptx -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_PTX_ARCHITECTURE=80
./scripts/generate_sass.sh build/ptx matmul_bench
```

生成结果默认在：

```text
build/ptx/cubin/matmul_bench.cubin
build/ptx/sass/matmul_bench.sass
```

你也可以直接用 CMake target：

```bash
cmake --build build/ptx --target matmul_bench_cubin
cmake --build build/ptx --target vector_add_bench_cubin
```

脚本行为：

- 优先使用 `nvdisasm -g` 生成带源码位置信息的 SASS
- 如果没有 `nvdisasm`，回退到 `cuobjdump --dump-sass`
- 适合把源码、PTX、SASS 和 `ncu` 报告对着看

## 一键检查工作流

如果你想把 benchmark、PTX、SASS、Nsight Compute 串起来，直接用：

```bash
./scripts/inspect_kernel.sh build/profile matmul_bench \
  -- --m 2048 --n 2048 --k 2048 --tile 16 --warmup 10 --iters 20
```

这个脚本会按顺序尝试：

- build benchmark
- 运行 benchmark
- 生成 PTX
- 生成 SASS
- 如果机器上有 `ncu`，再跑 Nsight Compute

常见变体：

```bash
./scripts/inspect_kernel.sh build/ptx matmul_bench --skip-ncu -- --m 1024 --n 1024 --k 1024
./scripts/inspect_kernel.sh build/profile matmul_bench --skip-sass -- --m 2048 --n 2048 --k 2048
```

说明：

- `--` 后面的参数会原样传给 benchmark
- 默认要求对应 build 目录已经 configure 过
- 如果需要，也可以先加 `--configure` 或 `--configure-preset profile`

## 使用 Nsight Compute

最简单的方式：

```bash
./scripts/profile_ncu.sh ./build/release/bin/vector_add_bench --elements 16777216 --iters 50
```

profile `matmul` 时推荐这样起步：

```bash
./scripts/profile_ncu.sh ./build/profile/bin/matmul_bench \
  --m 2048 --n 2048 --k 2048 --tile 16 --warmup 10 --iters 20
```

脚本会：

- 检查 `ncu` 是否存在
- 创建 `results/ncu/`
- 输出 `.ncu-rep` 报告
- 额外导出一份文本摘要

你也可以自己直接运行：

```bash
ncu --set full --kernel-name regex:vector_add_kernel \
  --launch-skip 20 \
  --launch-count 1 \
  ./build/release/bin/vector_add_bench --elements 16777216 --warmup 20 --iters 50
```

推荐习惯：

- benchmark 程序内部保留 warmup
- `ncu` 再用 `--launch-skip` 跳过早期 kernel launch
- 用 `-lineinfo` 构建，这样 source view 更清晰
- 先看 `SpeedOfLight`、occupancy、memory throughput，再决定往计算还是访存优化
- 看 `matmul` 时顺手关注 shared memory 利用、SM busy、tensor core 是否未被使用

## 如何新增自己的 kernel benchmark

1. 在 `benchmarks/` 新建一个 `.cu`
2. 也可以直接复制 `benchmarks/benchmark_template.cu`
3. 复用 `include/cuda_benchmark.cuh` 里的计时与统计工具
4. 在 `CMakeLists.txt` 里新增一行：

```cmake
add_cuda_benchmark(my_kernel_bench benchmarks/my_kernel_bench.cu)
```

5. 重新构建并运行：

```bash
cmake --build --preset release
./build/release/bin/my_kernel_bench
```

## Docker 方式

如果目标机器安装了 NVIDIA Container Toolkit：

```bash
docker build -t cuda-benchmarks .
docker run --rm -it --gpus all -v "$(pwd)":/workspace cuda-benchmarks
cmake --preset release
cmake --build --preset release
./build/release/bin/vector_add_bench
```

## 后续建议

- 增加 `matmul/softmax/reduction` 等真实 kernel case
- 当前已经有 shared-memory tiled `matmul` baseline，可以继续加 naive / vectorized / WMMA 版本做横向对比
- 按 kernel 类型拆子目录，例如 `benchmarks/memory/`、`benchmarks/reduction/`
- 增加一个统一脚本批量跑 benchmark 并落盘 CSV
- 后面如果你愿意，我可以继续帮你补一个更系统的 `results/` 输出格式和 roofline-friendly 指标汇总
