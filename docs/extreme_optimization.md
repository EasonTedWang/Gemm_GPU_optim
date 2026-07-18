# GEMM 极致优化路线

## 1. 目标与验收口径

这个阶段不再以“比上一个 kernel 快”作为唯一目标，而使用三层验收：

1. 正确性：规则 shape、非整除 shape、固定随机种子都必须通过 CPU reference。
2. 绝对性能：记录 kernel time、GFLOP/s、寄存器、shared memory 和 occupancy。
3. 相对性能：同精度、同布局、同 shape 下与 cuBLAS 比较，计算 `custom / cuBLAS`。

FP32 CUDA Core kernel 只和 FP32 cuBLAS 比。FP16/BF16/TF32 Tensor Core kernel 必须建立各自的 cuBLASLt 基线和误差门槛，不能拿 Tensor Core 吞吐除以 FP32 CUDA Core 峰值。

## 2. 当前高性能 CUDA Core kernel

`code/05_optimized/gemm_optimized.cu` 使用以下层级：

- CTA tile：128x128
- warp tile：32x64，单 CTA 共 8 个 warp
- thread tile：8x8，每线程保留 64 个 FP32 accumulator
- K stage：默认 16，同时保留 K=8 调优版本
- global load：A/B 各使用 128-bit `float4` 加载
- shared layout：A 转置写入，B 使用每 8 列 padding/swizzle
- pipeline：双 shared-memory buffer，预取下一 K stage 后计算当前 stage
- epilogue：规则 shape 使用 128-bit store，任意 shape 使用带边界检查的安全路径

RTX 5080 / CUDA 13.3 编译资源：

| Variant | Registers/thread | Static shared memory | Local spill | Active blocks/SM |
| --- | ---: | ---: | ---: | ---: |
| CTA_K=8 | 127-128 | 17 KB | 0 | 2 |
| CTA_K=16 | 127-128 | 34 KB | 0 | 2 |

2026-07-18 的一次同批次对比如下：

| Shape | tiled4x4 | optimized K=8 | optimized K=16 | cuBLAS | K=16/cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024³ | 10.55 TFLOP/s | 16.06 TFLOP/s | 18.54 TFLOP/s | 26.45 TFLOP/s | 70.1% |
| 2048³ | 16.18 TFLOP/s | 19.13 TFLOP/s | 20.50 TFLOP/s | 32.11 TFLOP/s | 63.8% |

GPU 当时同时承担桌面负载，因此数字只用于确认优化方向，正式结论要在空闲 GPU 上重复采样。

### FP32 异步拷贝结论

`gemm_optimized_async_cuda` 实现了双缓冲 `cp.async`、A row-dependent XOR layout 和 B chunk-level XOR swizzle。它通过了 aligned/boundary 正确性，但在 1024/2048 方阵上比软件预取主线慢约 15-30%。原因是额外异步指令、地址变换和 128 registers/thread 抵消了延迟隐藏收益。因此它作为可复现负结果保留，默认 FP32 kernel 仍是 `CTA_K=16` 软件预取版本。

## 3. 当前高性能 Tensor Core kernel

`code/05_optimized/gemm_tensor_tiled.cu` 已实现：

- FP16 A/B、FP32 accumulator/output
- CTA/warp/instruction 三级 tiling
- 4/8 warp 可配置 CTA，每 warp 复用 A/B fragment 计算多个 16x16 输出 tile
- A/B 16-byte `cp.async`、双 shared-memory buffer
- SM90+ 二维 TMA、`mbarrier` phase 复用和两/三级流水候选
- aligned 快路径，以及 M/N 为 16 倍数时支持任意 K 的边界路径
- 64x64x16、128x128x16、128x64x16、三级流水和 TMA 配置矩阵

当前选择规则由 `scripts/run_best_tensor.sh` 实现。满足 TMA 对齐时：

- 128x64 CTA 网格不超过 84 blocks 且 K 可被 32 整除：四-warp 2x2、128x64x32 两级 TMA
- 多 wave 且 M/N 至少为 32 的极端 tall：八-warp 128x64x16 两级 TMA
- 其余可被 128x64x16 整除的 shape：四-warp 2x2、128x64x16 两级 TMA
- 只能被 128x128x16 整除：128x128x16 两级 TMA
- 不满足 TMA fast path 时：回退到原有 `cp.async`/boundary-safe 规则

| Shape | 原始 WMMA | tiled 64x64x16 | tiled 128x128x16 | cuBLAS Tensor | 最佳/cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024³ | 19.22 TFLOP/s | 29.56 TFLOP/s | 42.54 TFLOP/s | 82.22 TFLOP/s | 51.7% |
| 2048³ | - | 38.04 TFLOP/s | 61.96 TFLOP/s | 91.42 TFLOP/s | 67.8% |
| 4096³ | - | - | 67.81 TFLOP/s | 95.01 TFLOP/s | 71.4% |

128x128x32 减少了 barrier 次数，但在 2048/4096 上因 114 registers/thread 和 32 KB shared memory 未能胜过 K=16，因此只保留为调优证据。

三级 128x128x16 在 1024/2048 上比两级流水提高约 5.7%/2.3%，但在 4096 上下降约 3%，因此只用于中等输出 tile 数。

### SM120 TMA 主线

CUDA 13.3 的 PTX wrapper 明确将 `tcgen05`/TMEM 限定在 SM100/103/110 family，RTX 5080 的 SM120 不支持该指令族；普通 `cp.async.bulk.tensor` 与 `mbarrier` 则支持 SM90+。因此本仓库的 SM120 原生路径采用 TMA 搬运加 WMMA 计算，而不是不可执行的 `tcgen05`。

两级 TMA 以单线程发起 A/B 二维 tile 搬运，其余 warp 专注 WMMA。相较三级 `cp.async`，128x128 配置从 104 降至 96 registers/thread；128x64 配置只需 56 registers/thread、12,304 B shared memory，可驻留 4 blocks/SM。

| Shape | 原 `cp.async` 主线 | 最佳 TMA | 提升 |
| --- | ---: | ---: | ---: |
| 1024³ | 41.87 TFLOP/s | 67.23 TFLOP/s (128x64 2x2) | 60.6% |
| 2048³ | 62.28 TFLOP/s | 82.73 TFLOP/s (128x64 2x2) | 32.8% |
| 4096³ | 70.11 TFLOP/s | 102.56 TFLOP/s (128x64 2x2) | 46.3% |

三级 TMA 也通过正确性，但在三个方阵尺寸都不及两级版本；它作为 stage 深度对照保留，不进入自动分派。

### TMA K-stage autotune

K=32 将每个 K 循环的 TMA 发射和 `mbarrier` 周期减半，但 128x64 配置会从 56 增至 62 registers/thread、从 12,304 B 增至 24,592 B shared memory，并使 active blocks/SM 从 4 降到 3。因此它只适合 CTA 数不足以填满 GPU 的场景。

RTX 5080 有 84 个 SM。实测 128x64 网格为 32 blocks 的 512 方阵、72 blocks 的 768 方阵，K=32 分别提升约 31% 和 40%；网格增至 98 blocks 的 896 方阵后反而慢约 10%。非方阵的 4096x64x4096 和 128x4096x4096 分别提升约 7.8% 和 9.4%，验证了按首个 SM wave 分派比按矩阵边长分派更可靠。

四-warp 2x2 与 K=32 组合后，768 方阵进一步达到 46.91 TFLOP/s，4096x64x4096 达到 20.39 TFLOP/s。512 方阵与八-warp K=32 基本持平，因此单-wave 统一选择组合版本。

128x128x32 TMA 在 2048/4096 方阵仅达到 59.93/74.16 TFLOP/s，低于 K=16 主线；该目标作为同步与资源权衡的负对照保留。

### Warp topology autotune

128x64 CTA 的默认 4x2 warp grid 使用 8 warps，每 warp 计算 32x32。四-warp 4x1 和 2x2 都将每 warp 输出扩大到 8 个 WMMA fragment，并把每个 CTA 的 A/B fragment load 总数从 32 降到 24。两者均使用 95 registers/thread、12,304 B shared memory，可驻留 5 blocks/SM。

4x1 在主 shape 上没有稳定收益；2x2 通过减少 row-major B 的重复 fragment load，在 1024/2048/4096 方阵达到 67.23/82.73/102.56 TFLOP/s。4096 长重复复测为 92.79 TFLOP/s，同批次 cuBLASLt 为 93.28 TFLOP/s，达成率 99.5%。

2x2 在 4096x256x4096、256x4096x4096、128x8192x4096 上相对八-warp TMA 提升约 32.3%、8.8%、9.0%；8192x128x4096 则下降约 9.2%。这说明当前 row-major 布局需要为极端 tall 保留八-warp高并行度版本。

### cuBLASLt 上限

`gemm_cublaslt_tensor` 使用 row-major layout、8 个 heuristic 候选和最多 64 MB workspace。它比基础 `cublasGemmEx` 更快，成为新的 Tensor 主基线：

| Shape | 当前最佳手写 | cuBLASLt | 达成率 |
| --- | ---: | ---: | ---: |
| 1024³ | 67.23 TFLOP/s | 86.48 TFLOP/s | 77.7% |
| 2048³ | 82.73 TFLOP/s | 98.05 TFLOP/s | 84.4% |
| 4096³ | 102.56 TFLOP/s | 104.46 TFLOP/s | 98.2% |

不同批次会受桌面负载和频率波动影响，表格用于配置选择，不作为稳定峰值声明。

### 矩形与 skinny shape

128x64x16 三级配置使用 62 registers/thread、18 KB shared memory、4 blocks/SM。它不仅适合 tall，也因为当前 row-major B 和 warp grid 的非对称性而胜过镜像的 64x128 配置。

| Shape | 128x128 s3 | 128x64 s3 | cuBLASLt | 128x64 提升 |
| --- | ---: | ---: | ---: | ---: |
| 4096x256x4096 | 42.27 TFLOP/s | 48.64 TFLOP/s | 106.35 TFLOP/s | 15.1% |
| 256x4096x4096 | 38.78 TFLOP/s | 49.11 TFLOP/s | 99.74 TFLOP/s | 26.6% |
| 8192x128x4096 | 29.28 TFLOP/s | 31.08 TFLOP/s | 101.67 TFLOP/s | 6.1% |
| 128x8192x4096 | 42.34 TFLOP/s | 47.03 TFLOP/s | 96.14 TFLOP/s | 11.1% |

64x128 配置通过正确性测试，但在这些 row-major shape 上均未胜出，因此只保留为布局实验。

两级 128x64 TMA 在同一组 shape 上进一步达到 57.49、59.93、61.61、62.39 TFLOP/s，相对旧 128x64 `cp.async` 分别提升 18.2%、22.0%、98.2%、32.7%。极窄 shape 的大幅提升来自 copy 指令减少、寄存器下降和 active blocks/SM 从 4 个稳定维持到计算阶段。

四-warp 2x2 TMA 在同一组 shape 上达到 76.06、65.22、55.97、68.03 TFLOP/s。自动分派仅在极端 tall 的第三个 shape 保留八-warp版本，其余采用 2x2。

## 4. 固定优化循环

每次只改变一个可解释变量，并严格执行：

```text
环境检查
  -> aligned + boundary correctness
  -> 512/1024/2048/4096 square benchmark
  -> tall/wide/skinny benchmark
  -> cuBLAS 达成率
  -> Nsight Compute 定位瓶颈
  -> 保留或回退该变量
```

建议固定以下矩阵集合：

| 类型 | M x N x K |
| --- | --- |
| 小矩阵 | 128x128x128, 256x256x256 |
| 主吞吐 | 1024³, 2048³, 4096³ |
| 非整除 | 1000x1003x997, 127x131x33 |
| Tall-skinny | 4096x256x4096 |
| Wide | 256x4096x4096 |
| Small-K | 4096x4096x64 |

正式 benchmark 要求 GPU 空闲、固定功耗/频率策略、至少 10 次 warmup 和 50 次计时。结果至少记录中位数；需要发表或长期比较时同时记录 P10/P90。

## 5. Nsight Compute 决策树

使用：

```bash
./scripts/profile_optimized.sh 2048
./scripts/profile_tensor.sh 2048
```

重点观察：

- DRAM/L2 吞吐高、FMA 低：扩大数据复用、改善 CTA 调度或做 split/persistent tile。
- shared-memory replay 高：调整 A/B layout、padding 或 XOR swizzle。
- eligible warps 低、barrier stall 高：增加 K stage、异步拷贝和多级 pipeline。
- local load/store 非零：发生寄存器 spill，需要缩小 thread tile 或改变 launch bounds。
- FP32 pipe 接近饱和：CUDA Core 路线接近本 shape 上限，转向 epilogue/fusion 或 Tensor Core。
- 小矩阵 cuBLAS 领先明显：减少 launch/初始化开销，考虑 grouped/batched/persistent kernel。

## 6. 下一阶段优化顺序

### P0：测量可信度

- 保持 `sm_120` 架构，不允许旧 cache 回退到 `sm_75`。
- cuBLAS 必须与 nvcc 来自同一个 Toolkit；当前 CMake 使用 `NO_DEFAULT_PATH` 绑定 Toolkit 内库。
- 增加 CSV/JSON 结果归档和中位数统计。
- GPU performance counters 开放后保存完整 `.ncu-rep`。
- WSL2 环境需要在 Windows NVIDIA Control Panel 的 Developer 设置中允许非管理员访问 GPU performance counters；Linux 侧没有 `/proc/driver/nvidia/params` 可修改。

### P1：CUDA Core 主线

- 使用 Nsight 数据解释经典 `cp.async` 为何慢于软件预取，再决定是否进入 TMA 路线。
- 对 128x128x8、128x128x16、128x64x16、64x128x16 做离线 autotune。
- 比较 8x8、8x4、4x8 thread tile，约束零 spill 和足够 active warps。
- 使用 XOR swizzle 让 128-bit shared store 与无 bank-conflict read 同时成立。
- 增加 alpha/beta 和向量化 epilogue，为 bias/ReLU/GELU fusion 做准备。

目标不是固定 TFLOP/s，而是在主 shape 上稳定达到 FP32 cuBLAS 的 75% 以上，并解释剩余差距。

### P2：Tensor Core 主线

- 用 Nsight 计数器解释两级 TMA 胜过三级 TMA 的 barrier/occupancy 原因。
- 扩展 cuBLASLt baseline，记录算法 ID、tile、split-K 和 workspace 属性。
- 继续优化 SM120 可执行的 TMA + WMMA 路线；`tcgen05` 仅在支持的目标架构上另建实现。
- 为 BF16/TF32 建立独立 kernel、误差标准和库基线。
- 目标是在 4096³ 上稳定达到同精度 cuBLAS Tensor 的 80% 以上。

### P3：真实工作负载

- 支持转置、leading dimension、batch、alpha/beta。
- 为 small-K、skinny 和超大矩阵分别选择 kernel，不追求一个配置覆盖所有 shape。
- 对大 K/M/N 评估 split-K、persistent CTA 和 stream-K。
- 融合 epilogue，减少 GEMM 后额外 global-memory round trip。

## 7. 完成定义

“极致优化”不是某一个 kernel 的终点，而是一套持续可验证的系统。仓库达到下一里程碑需要同时满足：

- 所有正确性测试通过，含非整除 shape。
- CUDA Toolkit、目标架构和基线库版本一致。
- 主 shape 有稳定的 cuBLAS 达成率，不使用单次最好值。
- 每次保留的优化都有 profiler 或重复 benchmark 证据。
- CUDA Core 与 Tensor Core 使用各自正确的精度和峰值口径。
