# 02 Tiled GEMM Optimization Notes

本文记录 `02_tiled` 阶段的优化思路、当前性能、为什么性能低、以及下一步代码怎么写。这个阶段的目标不是一次写出最终 GEMM，而是把 shared-memory tiling 的性能边界摸清楚。

## 当前硬件和评价口径

本仓库默认使用 RTX 5080 16GB profile 作为 GPU 参考：

- CUDA cores: 10752
- Boost clock: 2.62 GHz
- FP32 peak: `10752 * 2 * 2.62 = 56.34 TFLOP/s`
- Memory bandwidth: 960 GB/s

框架打印的 `fp32_efficiency` 是：

```text
measured_gflops / 56340.48 * 100%
```

注意：Tensor Core 有自己的峰值口径；这里的 tiled FP32 kernel 用 FP32 CUDA core 峰值做参照是合理的。

## 已有版本

### tiled16 baseline

代码：`code/02_tiled/gemm_tiled.cu`

- Block tile: 16x16 C
- Thread block: 16x16 = 256 threads
- 每个线程计算 1 个 C 元素
- 每轮 K tile 从 global memory 读入 16x16 A 和 16x16 B 到 shared memory

核心形态：

```cpp
constexpr int TILE_DIM = 16;
__shared__ float sharedA[TILE_DIM][TILE_DIM];
__shared__ float sharedB[TILE_DIM][TILE_DIM];

int row = blockIdx.y * TILE_DIM + threadIdx.y;
int col = blockIdx.x * TILE_DIM + threadIdx.x;

for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
    sharedA[threadIdx.y][threadIdx.x] = ...;
    sharedB[threadIdx.y][threadIdx.x] = ...;
    __syncthreads();

    for (int k = 0; k < TILE_DIM; ++k) {
        sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
    }

    __syncthreads();
}
```

### tiled32 one-output-per-thread

代码：`code/02_tiled/gemm_tiled32.cu`

- Block tile: 32x32 C
- Thread block: 32x32 = 1024 threads
- 每个线程仍然计算 1 个 C 元素
- Global memory 复用更高，但 1024 threads/block 太重

这个版本是故意保留的学习对照：它证明“只把 tile 变大”不一定更快。

### tiled2x2 register reuse

代码：`code/02_tiled/gemm_tiled2x2.cu`

- Block tile: 32x32 C
- Thread block: 16x16 = 256 threads
- 每个线程计算 2x2 个 C 元素
- Shared memory tile 仍然是 32x32 A 和 32x32 B
- 每个线程把 2 个 A 值和 2 个 B 值临时放进寄存器，再做 4 个 FMA

核心形态：

```cpp
constexpr int BLOCK_TILE_M = 32;
constexpr int BLOCK_TILE_N = 32;
constexpr int BLOCK_TILE_K = 32;
constexpr int THREAD_TILE_M = 2;
constexpr int THREAD_TILE_N = 2;

float accum[THREAD_TILE_M][THREAD_TILE_N] = {};

for (int k = 0; k < BLOCK_TILE_K; ++k) {
    float aFrag[THREAD_TILE_M];
    float bFrag[THREAD_TILE_N];

    for (int i = 0; i < THREAD_TILE_M; ++i) {
        aFrag[i] = sharedA[threadIdx.y * THREAD_TILE_M + i][k];
    }
    for (int j = 0; j < THREAD_TILE_N; ++j) {
        bFrag[j] = sharedB[k][threadIdx.x * THREAD_TILE_N + j];
    }

    for (int i = 0; i < THREAD_TILE_M; ++i) {
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            accum[i][j] += aFrag[i] * bFrag[j];
        }
    }
}
```

这一步的本质不是“更大的 tile”，而是“一个线程做多个输出，使 A/B 从 shared memory 读出后能在寄存器里被多个 FMA 复用”。

### tiled4x4 larger thread tile

代码：`code/02_tiled/gemm_tiled4x4.cu`

- Block tile: 64x64 C
- Thread block: 16x16 = 256 threads
- 每个线程计算 4x4 个 C 元素
- K tile 使用 16，控制 shared memory 和寄存器压力
- shared memory 对 A/B tile 做 +1 padding，降低典型列访问 bank conflict 风险

这个版本延续 `tiled2x2` 的思路，但把每线程输出从 4 个提升到 16 个。收益主要来自 shared memory 数据进入寄存器后能服务更多 FMA；代价是 accumulator 更多、寄存器压力更高，所以不能只看 tile 变大，还要观察 occupancy 和调度效率。

## 为什么 tiled16 只有约 7%

以 16x16 tile 为例，忽略 C 写回，每个 K tile 的粗略计算是：

```text
FLOPs = 2 * TILE_DIM^3
Global bytes = 2 * TILE_DIM^2 * sizeof(float)
Arithmetic intensity = FLOPs / bytes = TILE_DIM / 4 FLOP/byte
```

所以：

- TILE=16: `4 FLOP/byte`
- TILE=32: `8 FLOP/byte`

如果只看 960 GB/s 显存带宽，TILE=16 的粗略 bandwidth roof 是：

```text
960 GB/s * 4 FLOP/byte = 3.84 TFLOP/s
```

实测 tiled16 在 512/1024 的范围约 4.1-4.6 TFLOP/s，已经和这个粗略 roof 很接近。它并不是“完全没优化”，而是这个写法的上限本来就不高。

更关键的是，tiled16 每个线程每个 k 只做 1 个 FMA：

```text
load sharedA + load sharedB + 1 FMA
```

也就是说，global memory 被 shared memory 优化了一次，但 shared memory 到寄存器的复用还不够。GPU 算力很强，单个线程只做一个输出会让指令、shared memory load、同步开销占很大比例。

## 为什么 tiled32 不一定更快

把 tile 从 16 改成 32，理论上 global-memory arithmetic intensity 从 4 提高到 8。但 `gemm_tiled32.cu` 使用 32x32 = 1024 threads/block：

- block 太大，调度弹性下降
- 每个线程仍然只计算 1 个输出
- 每个 FMA 仍然需要从 shared memory 取 2 个数
- 只减少 global memory 压力，没有改善 shared-memory/register 复用

因此 tiled32 在本机 512/1024 测试中没有超过 tiled16。

## 当前实测

测试命令：

```bash
./out/gemm_tiled_cuda 512 512 512 --verify --warmup 5 --repeat 20
./out/gemm_tiled32_cuda 512 512 512 --verify --warmup 5 --repeat 20
./out/gemm_tiled2x2_cuda 512 512 512 --verify --warmup 5 --repeat 20
./out/gemm_tiled4x4_cuda 512 512 512 --verify --warmup 5 --repeat 20

./out/gemm_tiled_cuda 1024 1024 1024 --no-verify --warmup 5 --repeat 20
./out/gemm_tiled32_cuda 1024 1024 1024 --no-verify --warmup 5 --repeat 20
./out/gemm_tiled2x2_cuda 1024 1024 1024 --no-verify --warmup 5 --repeat 20
./out/gemm_tiled4x4_cuda 1024 1024 1024 --no-verify --warmup 5 --repeat 20
```

结果会随驱动、功耗、温度和后台负载波动。当前一次实测如下：

| Kernel | Shape | avg_ms | GFLOP/s | FP32 efficiency |
| --- | ---: | ---: | ---: | ---: |
| tiled16 | 512^3 | 0.0657 | 4084.43 | 7.2496% |
| tiled32 | 512^3 | 0.0762 | 3520.93 | 6.2494% |
| tiled2x2 | 512^3 | 0.0345 | 7785.25 | 13.8182% |
| tiled4x4 | 512^3 | 0.0333 | 8072.96 | 14.3289% |
| tiled16 | 1024^3 | 0.4689 | 4579.84 | 8.1289% |
| tiled32 | 1024^3 | 0.4928 | 4357.92 | 7.7350% |
| tiled2x2 | 1024^3 | 0.1751 | 12261.47 | 21.7632% |
| tiled4x4 | 1024^3 | 0.1652 | 13002.07 | 23.0777% |

正确性：

- tiled16 512 verify: PASSED
- tiled32 512 verify: PASSED
- tiled2x2 512 verify: PASSED
- tiled2x2 64x65x66 非整除 shape verify: PASSED
- tiled4x4 512 verify: PASSED
- tiled4x4 32x33x34 非整除 shape CTest verify: PASSED

## tiled 阶段最高能到多少

这里要先定义“tiled 阶段”：

1. 如果限定为 `每个线程只计算 1 个输出` 的 shared-memory tiled，那么上限很低。本机目前 tiled16/tiled32 大概在 6-8% FP32 peak。继续微调 tile size 可能有小幅波动，但很难质变。
2. 如果允许 `每个线程计算多个输出`，但仍然只使用 shared-memory block tile，不进入 warp-level MMA / Tensor Core，那么 tiled2x2/tiled4x4 已经到 13-23% FP32 peak。继续做 vectorized load、减少同步和处理 bank conflict，可能把这个阶段推进到 20-30% 左右。
3. 想接近现代 GEMM 的高效率，必须进入 register tiling + warp-level tiling + 更精细的 memory pipeline。想大幅超过 FP32 CUDA core 路线，则进入 Tensor Core。

所以当前最重要的结论是：

```text
shared memory tiling 解决的是 global memory reuse；
每线程多个输出解决的是 shared memory -> register reuse；
更高性能还需要 warp/block 层级的结构化 tiling。
```

## 下一步怎么提高

建议下一轮从 `tiled4x4` 继续，而不是从 `tiled32` 继续：

1. 先 profile `tiled4x4`
   - 观察 register count、occupancy、shared memory throughput
   - 确认 4x4 的瓶颈是寄存器压力、shared memory 访问，还是 global load 指令数
2. 尝试 `BLOCK_TILE_K=8/32`
   - `K=8` 可能降低 shared memory 和寄存器压力
   - `K=32` 可能提高 global-memory reuse，但会增加 shared memory 占用
3. 加 vectorized global load
   - 用 `float4` 或让连续线程加载连续地址
   - 目标是减少 load 指令，提高 memory transaction 效率
4. 分析 shared memory bank conflict
   - 尤其是 B tile 的访问方式
   - 可尝试 padding：`sharedB[BLOCK_TILE_K][BLOCK_TILE_N + 1]`
5. 记录每一步
   - 每次只改一个变量
   - 每次跑 verify 和固定 shape benchmark
   - 把 GFLOP/s、efficiency、结论写进文档

## 启发

- 不要把 “tile 大” 等同于 “更快”。tile 大只提高 global reuse，但可能降低 occupancy 和调度弹性。
- GEMM 的性能来自层级复用：global -> shared -> register。当前 tiled16 只做了第一层。
- 每线程多个输出是从 naive tiled 走向高性能 GEMM 的关键转折点。
- 性能优化必须和正确性测试绑定。`tiled2x2` 和 `tiled4x4` 都已经覆盖普通 shape 和非整除 shape 的正确性。
- 当某个版本没有变快，它也有价值：`tiled32` 说明瓶颈已经从 global memory 转向 shared-memory/register/调度层面。
