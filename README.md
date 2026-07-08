# Gemm_GPU_optim

利用 CUDA/AVX 等硬件从零开始优化 GEMM，借由 CODEX 来辅助学习。

## 当前目标

这个仓库的核心目标是建立一套可反复实验的 GEMM 学习环境：每个 kernel 都要能验证正确性，也要能输出基本性能指标，后续再逐步推进 CPU、CUDA、AVX、Tensor Core 等优化版本。

## 项目结构

- `CMakeLists.txt` 根级 CMake 工程
- `docs/gemm_learning_plan.md` GEMM 优化学习路线
- `docs/02_tiled_optimization.md` tiled GEMM 阶段优化笔记
- `code/origin_gemm.c` CPU reference / baseline
- `code/common/` 公共实验工具
  - `matrix_utils.*` 参数解析、随机矩阵、CPU reference、误差比较、GFLOPS 输出
  - `cuda_utils.cuh` CUDA error check 和 kernel 计时器
- `code/01_naive/` Naive CUDA GEMM
- `code/02_tiled/` Shared-memory tiled CUDA GEMM
  - `gemm_tiled_cuda`: 16x16 baseline
  - `gemm_tiled32_cuda`: 32x32 one-output-per-thread 对照版本
  - `gemm_tiled2x2_cuda`: 32x32 block tile + 每线程 2x2 输出
  - `gemm_tiled4x4_cuda`: 64x64 block tile + 每线程 4x4 输出
- `code/03_warp/` Register-tiled CUDA GEMM
- `code/04_tensor_core/` WMMA Tensor Core GEMM

## 构建

```bash
./build_all.sh
```

或手动执行：

```bash
mkdir -p build
cd build
cmake .. -DENABLE_CUDA=ON
cmake --build . --target all_examples
```

如果没有 CUDA Toolkit：

```bash
cmake .. -DENABLE_CUDA=OFF
cmake --build . --target all_examples
```

## 统一运行参数

所有示例都支持同一组基础参数：

```bash
./out/gemm_naive_cuda [M N K] [options]
```

常用选项：

- `--verify`：用 CPU reference 做全矩阵正确性校验
- `--warmup N`：计时前预热次数，默认 3
- `--repeat N`：计时循环次数，默认 10
- `--seed N`：固定随机种子，默认 2026
- `--no-benchmark`：只运行一次，不输出计时
- `-h` / `--help`：查看帮助

示例：

```bash
./out/origin_gemm 256 256 256 --verify --repeat 1 --warmup 0
./out/gemm_naive_cuda 256 256 256 --verify --repeat 10 --warmup 3
./out/gemm_tiled_cuda 256 256 256 --verify --repeat 10 --warmup 3
./out/gemm_tiled32_cuda 256 256 256 --verify --repeat 10 --warmup 3
./out/gemm_tiled2x2_cuda 256 256 256 --verify --repeat 10 --warmup 3
./out/gemm_tiled4x4_cuda 256 256 256 --verify --repeat 10 --warmup 3
./out/gemm_warp_cuda 256 256 256 --verify --repeat 10 --warmup 3
./out/gemm_tensor_core 256 256 256 --verify --repeat 10 --warmup 3
```

输出中的 `avg_ms` 是 kernel 平均耗时，`gflops` 使用 `2*M*N*K / time` 计算。CUDA 示例的计时只覆盖 kernel，不包含 Host/Device 数据拷贝。

当前默认 GPU profile 是 `NVIDIA GeForce RTX 5080 16GB`：

- CUDA cores: 10752
- Boost clock: 2.62 GHz
- FP32 peak: `10752 * 2 * 2.62 = 56.34 TFLOP/s`
- Memory bandwidth: 960 GB/s

CUDA benchmark 输出会额外打印 `fp32_peak` 和 `fp32_efficiency`，其中 `fp32_efficiency = measured_gflops / fp32_peak_gflops * 100%`。Tensor Core 示例暂时也打印这个 FP32 峰值参照，后续可以再补独立的 Tensor Core 峰值口径。

## 测试

构建后运行：

```bash
ctest --test-dir build --output-on-failure
```

当前测试会覆盖 CPU baseline、naive CUDA、tiled CUDA、register-tiled CUDA 和 Tensor Core 的小尺寸正确性冒烟，也会运行 512x512x512 的 CUDA 性能冒烟。

如果要在测试过程中直接看到每个 kernel 的 `avg_ms`、`gflops` 和 `fp32_efficiency`：

```bash
ctest --test-dir build --verbose
```

也可以使用项目内的 verbose 测试 target：

```bash
cmake --build build --target test_gemm_verbose
```

## 学习计划

完整学习路线见 [docs/gemm_learning_plan.md](docs/gemm_learning_plan.md)。建议按以下顺序推进：

1. 理解 GEMM 数学定义与 CPU reference
2. 跑通 naive CUDA 并观察访存瓶颈
3. 引入 shared memory tiling
4. 引入每线程多输出元素、register tiling、warp/block tile 设计
5. 学习 Tensor Core / WMMA
6. 逐步靠近 CUTLASS 风格的结构化 GEMM kernel
