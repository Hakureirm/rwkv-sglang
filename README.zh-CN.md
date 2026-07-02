# RWKV-7 (Goose) × sglang

[English](README.md) · **简体中文**

一个**生产级的 RWKV-7 在 [sglang](https://github.com/sgl-project/sglang) 上的实现**：
数值正确（对齐 BlinkDL `rwkv-lm` 参考，逐 token 精确）、自包含、可量化、可跨消费级与
数据中心 GPU 运行，并复用 sglang 原生的动态批处理（dynamic batching）、分块预填充
（chunked prefill）和一个**大小恒定的循环状态缓存**。

基于 **sglang v0.5.10.post1** 开发并验证（这是开发机 CUDA-12.9 驱动能跑的最新版本；
sglang `main` 需要 CUDA 13）。以**覆盖层（overlay）**形式交付（`sglang_overlay/`），
部署时叠加进已安装的 sglang —— 见[目录结构](#目录结构)。

> 说明：本项目把 RWKV-7 集成进 sglang 用于生产级 serving。目标：在精度上对齐 rwkv-lm 参考、
> 在速度/显存上对齐 albatross（跨各档 batch size）；复用 sglang 原生的动态批处理 + 分块预填充
> + 大小恒定的循环状态缓存；8/4-bit 量化不慢于 16-bit；覆盖消费级到数据中心的多种 GPU。

**快速跳转：** [📊 基准一览](#-基准一览) · [当前诚实标准](#当前诚实标准2026-07-01对标-blinkdl--albatross) · [设计目标与状态](#设计目标与状态) · [部署](#部署快速上手) · [目录结构](#目录结构) · [文档与决策](docs/)

---

## 为什么是 RWKV-7 × sglang

RWKV-7 是纯循环（RNN）架构，**每个 token 的状态是 O(1) 常量**，不随上下文长度增长。
对比之下，Transformer（含 Qwen3.5 的 KV 部分）的状态随序列长度线性膨胀。以 L24-D1024 为例：

| | 状态大小 |
|---|---|
| RWKV-7 | **1.62M（恒定）** |
| Qwen3.5 | 5.05M + 6.14×(T/1000) M（随上下文 T 增长） |

这意味着在**高并发、长上下文**场景下，RWKV-7 能在同样显存里塞下多得多的并发序列，
这正是它在 serving 上的结构性优势，也是本项目的切入点（wedge）。

---

## 📊 基准一览
独占 RTX 3090、可复现（≥7 次取中位）——方法与原始日志见 **[`bench/results/`](bench/results/)**
（服务规模 [`serving_scale/`](bench/results/serving_scale/)，同精度 [`comparison_clean.md`](bench/results/comparison_clean.md)，
精度 [`lm_eval.md`](bench/results/lm_eval.md)）。这是一个**服务引擎**交付，所以头牌放我们赢的服务轴；
albatross 的主场（同精度**单流**裸解码）随后完整摊开——不藏任何东西。

### 生产级服务引擎赢在哪 ✅

**1. 并发吞吐——填满 batch 可扩展约 50×**（1.5B，稳态解码 tok/s，RTX 3090）：
```
bsz   1  █░░░░░░░░░░░░░░░░░░░░░    166 tok/s
bsz  16  █████░░░░░░░░░░░░░░░░░  2,143
bsz  64  ████████████████░░░░░  6,445
bsz 128  █████████████████████  8,298
bsz 256  ████████████████████░  8,187   (算力受限，进入平台)
```

**2. 显存 O(1)——并发与上下文双恒定**（1.5B，nvidia-smi 峰值）：
| 扩展轴 | 基线 | 放大后 | Δ 峰值显存 |
|---|---|---|---|
| **并发** | bsz 1 = 12,420 MiB | **bsz 256** = 12,622 MiB | **+202 MiB** 承载 256 条并发 |
| **上下文** | 1K = 12,364 MiB | **64K** = 12,368 MiB | **+4 MiB**（上下文放大 64×） |

每条 RWKV-7 状态是固定 162 万元素的常量（**无 KV cache**），所以 256 条并发——在*任意*上下文长度下——
显存开销与一条几乎相同。KV-cache Transformer 的显存随 batch × 上下文增长，早就 OOM 了。解码保持
**O(1)/token**（无论上下文多长都是个位数 ms/step；TTFT 是 O(T)，任何模型都如此）。**7.2B 同样成立**：
上下文 1K→32K 峰值显存 **+0 MiB**；并发 bsz 1→64 解码 46.6→1,802.7 tok/s（38.7×）、仅 +308 MiB——
单张 24G 卡承载 64 路并发 7.2B（[`serving_scale/`](bench/results/serving_scale/)）。

**3. int8 (w8a8) 基本追平 albatross-fp16**——7.2B 上，我们的 int8 落在
**albatross-fp16 的 0.88–1.21×**（解码，bsz 1/8/32；即*跨精度*对比：我们的 int8 vs 它的 fp16），
同时权重字节 **−46%**。这是 albatross 没有的量化路径。

**3b. 手写 weight-only int8(w8a16):贪心逐-token 精确,且 bsz≤32 每档快于/持平 fp16**——24/24
精确(实践无损),解码 bsz 1/2/4/8/16/32 为 fp16 的 1.37×/1.31×/1.27×/1.06×/**1.12×/1.02×**
(227/392/732/1181/2513/3936 tok/s;bsz64 0.74×,如实报),与 int4 同款三核分派(GEMV / 逐行位一致
小-M GEMM / smem 内解量化 tensor-core GEMM),且 **全架构可跑**(按架构 JIT)——不同于 cutlass
w8a8(仅 sm80–90)。`RWKV_W8=1`;详见
[`docs/findings/0018`](docs/findings/0018-w8-weight-only.md)。

**3c. 手写 int4 在 bsz≤32 每档快于/持平 fp16**——1.5B 解码 bsz 1/2/4/8/16/32 分别为 fp16 的
1.56×/1.45×/1.35×/1.04×/**1.17×/1.03×**（259/435/773/1153/2620/3978 vs 166/300/574/1113/2243/3873
tok/s；bsz64 为 0.77×，如实报），由三个手写核分派：`gemv_w4_m1`、`gemm_w4_small`（逐行位一致）、
tensor-core `gemm_w4_tc`（smem 内解量化 + 确定性 split-K）。**7.2B**：bsz1
**102.8 tok/s = albatross-fp16（79.6）的 1.29×**（跨精度），样本贪心 **8/8 精确**，lambada
0.7161 vs bf16 0.7425（−2.64pt，RTN）——并已**在真 16 GB T4 上实测**：贪心 8/8 精确、
bsz1 32.9 tok/s、峰值显存仅 **6.7 GB**。详见 [`bench/results/w4/`](bench/results/w4/)。

**4. 精度逐-token 精确**——对 rwkv-lm 纯 numpy oracle 贪心逐-token 命中，0.1B / 1.5B / 7.2B
（fp16 + bf16，cuda-graph）；lm-eval 与 rwkv-lm **持平**（1.5B lambada 0.673 vs 0.671，MMLU 0.524 vs 0.511）。

**5. 全 GPU 系列可跑**——**10 种 GPU、7 个 SM 世代、Turing → Blackwell**（T4 / L4 / A10G /
A100-40/80 / L40S / H100 / H200 / **B200** / **RTX PRO 6000**）各卡真机实测：bf16 **全 10 卡
逐-token 精确**；手写 **int4 全 10 卡可跑且 bsz1 比 fp16 快**——从 Turing（不依赖 cp.async）到
Blackwell sm120（RTX PRO 6000 上 int4 bsz1 **1.41×** bf16），无需按架构改代码。峰值：**B200
预填充 103,022 tok/s、解码 7,213 tok/s** @bsz32。（int8 仅支持 sm80–90——上游 sgl-kernel cutlass
覆盖限制。）完整网格见 [`bench/results/multigpu.md`](bench/results/multigpu.md)。

### albatross 唯一领先的那条轴——完整摊开 🔬
**同精度 fp16、*单流*裸解码。** 这是 albatross 的主场：它是纯单流 mega-kernel，已达
**3090 显存带宽天花板的 ~92%**，而我们是完整的动态批服务引擎。我们照样公布每一个数字
（越高越接近它的裸核；`1.00×` = 持平；最佳配置 = in-place WKV + `RWKV_SPARSE_FFN=1` + `RWKV_FAST_LINEAR=1`）：
```
              ours / albatross-fp16 —— 同精度单流（decode tok/s）
7.2B  bsz1  ██████████████████░░░░  0.83×   (45.9 → 65.7 tok/s)
7.2B  bsz8  █████████████████░░░░░  0.84×
7.2B  bsz32 ██████████████░░░░░░░░  0.72×   (in-place WKV +24%)
1.5B  bsz1  █████████████░░░░░░░░░  0.66×
1.5B  bsz8  ██████████████████░░░░  0.90×   ← 最接近持平
1.5B  bsz32 ██████████████░░░░░░░░  0.70×
```
0.1B 各行（0.49–0.79×）此处**不列**——launch 受限的极小模型是最不具服务代表性的场景；完整数字见
[`comparison_clean.md`](bench/results/comparison_clean.md)。即便在这条对我们最不利的轴上，我们真正会去
服务的中/大模型也在 **0.66–0.90×**，靠三个手写贪心精确核补齐（in-place WKV + 稀疏 FFN + 融合 GEMV）。

**结论：** 真实服务里——**并发、显存、int8、精度**——RWKV-7 × sglang 赢；albatross 只在同精度单流裸解码
上领先，且只在它的带宽天花板处。

---

## 当前诚实标准（2026-07-01，对标 BlinkDL / albatross）

下面对标 albatross 的数字为独占 RTX 3090 的干净测量、可复现（`bench/results/comparison_clean.md`
与 `lm_eval.md`，取代早期与其他任务共卡的 `comparison.md`）；**跨卡扫测覆盖 8 种架构
T4 → H200、各卡真机实测**（`bench/results/multigpu.md`）。

- ✅ **正确性**：RWKV-7 **0.1B / 1.5B / 7.2B** 全部 **贪心逐-token 精确**（greedy-EXACT）
  对齐 numpy / `rwkv-lm` 参考（fp16 + bf16）；动态批（共享前缀 / 混合）同样精确。
- ✅ **精度 = 与 rwkv-lm 持平**（lm-eval，1.5B）：lambada 0.673 vs 参考 0.671，
  MMLU 0.524 vs 0.511。（7.2B 为 8-token 定长样本上的贪心精确；1.5B 为完整打分。）
- ⚖️ **同精度原始速度：手写 CUDA 已补上大半差距。** fp16 对 fp16 解码现为
  **0.49–0.90× albatross（全尺寸/全 bsz）**（原 0.46–0.85×），靠三个**不依赖 FLA、贪心精确、
  批不变**的手写核：
  - **in-place 索引 WKV 状态读写**（默认）：WKV 递归是唯一随 batch 增长的解码分量，现直接
    读写分页状态池（不再 gather/scatter），显著抬升**批处理/生产档**：7.2B bsz32 0.61→**0.72×**、
    1.5B bsz32 0.57→**0.70×**（约 +24%）。
  - **稀疏 sqrelu FFN**（`RWKV_SPARSE_FFN=1`）：`relu(k)²` 真实 **86–90% 为零**，手写 fp32
    累加 SpMV 跳过约 9/10 value 权重读取；**融合 fp16 GEMV**（`RWKV_FAST_LINEAR=1`）管
    r/k/v/o+key 投影（bsz1 档）。
  - 最佳组合：**7.2B bsz1 45.9→65.7 tok/s（0.58→0.83×）**，1.5B bsz8 **0.90×**。证据见
    `bench/results/{comparison_clean.md,best2,sparse_ffn}`。albatross 仍领先原始解码（整层
    mega-kernel，~92% 带宽峰值）；彻底追平需同款整层融合，会牺牲干净集成。
- ✅ **int8（w8a8，albatross 没有的特性）**：7.2B 上，我们的 int8 **基本追平**
  albatross-fp16（解码 **0.88–1.21×**，bsz 1/8/32；跨精度对比：我们的 int8 vs 它的 fp16），不是同精度对比。
- ✅ **显存**：循环状态是 O(1)/token，**随 batch 恒定**；albatross 的静态 B×T 在
  7.2B bsz32 时逼近 OOM。int8 把权重字节再砍约 **46%**（7.2B）。
- ✅ **多卡（8 种架构 Turing→Hopper）**：T4/L4/A10G/A100-40/A100-80/L40S/H100/H200 全部
  bf16 贪心精确、int4 全卡可跑且更快，无需按架构改代码（`bench/results/multigpu.md`）。
  ✅ **RWKV-7 执行路径不含 FLA**（自研 WKV 核；精确范围见下文"参考与口径"）。
- 🔜 **待办**：fp8；World 分词器 serving 打磨 + 上游 PR。

**定位（诚实）：** 我们在**精度上与 rwkv-lm 持平**（已验证），在 **显存 / int8 / 真实
serving**（动态批——albatross 没有）上**领先**，并用三个手写贪心精确核**补上了同精度原始速度的
大半差距**——现为 albatross 的 **0.49–0.90×**（全尺寸/全 bsz；7.2B bsz1 0.83×，1.5B bsz8 0.90×）。
最后一点需 albatross 那种整层 mega-kernel（牺牲干净集成，最好也只~持平）。

---

## 设计目标与状态

对本项目工程目标的诚实自评（2026-07-01），范围限定为一个 sglang 推理集成；
✅ 完成，◑ 部分，⬜ 未做 / 不在本项目范围。

| # | 目标 | 本交付的状态 |
|---|---|---|
| 1 | 跨 bsz 达到 albatross/RWKV-LM 性能 | ◑ 精度**持平** RWKV-LM；显存/int8/serving **领先**；同精度原始 fp16 解码 **0.49–0.90×（全尺寸/全 bsz）**（原 0.46–0.85×），靠 3 个手写贪心精确核（in-place WKV + 稀疏 FFN + 融合 GEMV）——`bench/results/comparison_clean.md` |
| 2 | 同量化下比 Qwen3.5 快（典型场景） | — **不在本项目范围**（一个 sglang 推理集成）：本交付对标 **albatross**（速度/显存）+ **RWKV-LM**（精度） |
| 3 | transformers 的 PEFT/RL 训练 | ⬜ 不在本项目范围（一个 sglang 推理集成） |
| 4 | 动态批 + 分块预填充 + 状态缓存 | ✅ sglang 原生动态批 + 分块预填充 + O(1) 循环状态池；◑ 前缀**复用** radix 暂自动关闭（状态尚不可前缀缓存——已记录为 `MambaRadixCache` 后续项） |
| 5 | Pascal+/AMD/Intel/国产；PP+TP；zero2/3；autotune | ◑ 10 种 GPU（Turing→Blackwell）贪心精确；**TP 在 2/4/8 卡、PP 在 2/4/8 段均贪心 24/24 精确（真 L4 集群实测）**（tp=1/pp=1 零回归；混合 tp×pp 有已记录的 open bug；W4/W8 暂限 tp=1；完整矩阵含每卡显存：[`bench/results/parallel/`](bench/results/parallel/)、[`docs/findings/0019`](docs/findings/0019-tp-pp-parallel.md)）；⬜ Pascal/AMD/Intel 未测，训练/autotune 不在范围 |
| 6 | w8 + w4，比 w16 快，老卡，Q\*_K_M 精度 | ✅ **w8（w8a8-int8）**——比 bf16 快（1.5B/7.2B 解码 +46–59%）、权重 −46%、7.2B 贪心精确；✅ **w4（手写 int4）**——**bsz≤8 每档都比 fp16 快**（1.5B 1.04–1.56×；7.2B bsz1 102.8 tok/s、样本贪心 8/8 精确、lambada 0.7161 vs 0.7425、总显存 9.8 GB），Turing→Hopper 全卡可跑；◑ Q\*_K_M 直接对表未做（我们的 GPTQ g64 @1.5B −3.34pt 为可比点） |
| 7 | 初步投机解码（RWKV 做 draft） | ⬜ 未做 |

已完整验证、最强的贡献是：**精确正确性（0.1B/1.5B/7.2B）**、**int8 速度/显存**、
**sglang 原生 serving**、**多卡**、**自研不含 FLA 的 WKV 核**，以及一次
**严格测量、诚实汇报的 CUDA 终局**（F0015）。

---

## 精度与速度：参考与口径（无 FLA）

- **精度基准 = BlinkDL `rwkv` pip 包 + 一份纯 NumPy 转写**的 RWKV-7 递推
  （`bench/oracle_numpy.py`，遵循 BlinkDL 的 `rwkv_v7_numpy.py`）。我们**不**用
  flash-linear-attention 作为精度参考。
- **速度/显存基准 = BlinkDL/albatross**，在我们自己的 3090 上重测（`bench/results/`）。
- **核策略（ADR-0004）：RWKV-7 路径上不依赖 `flash-linear-attention`（PyPI 包）。**
  （overlay 里被改动的**上游** sglang 文件仍保留 sglang 自己的 `…fla…` mamba/gated-delta
  导入，但 RWKV-7 从不触发它们；模型目录名/转换脚本名里的 `-fla` 指的是 fla-格式的
  **权重布局**，不是代码依赖。）

---

## 目录结构

- `sglang_overlay/` —— **交付主体**：新增 + 修改的 sglang 文件（模型、状态后端、config、
  接线），通过 `scripts/deploy.sh` 叠加进 sglang（rsync 覆盖，无需编译）。
- `tools/convert_rwkv7_blinkdl_to_fla.py` —— 把 BlinkDL `.pth` 转成 sglang 可加载的权重。
- `bench/` —— 精度基准（`oracle_numpy.py`）、门禁（`verify_m1d.py`、`verify_batch.py`）、
  吞吐（`throughput.py`、`run_clean_comparison.py`）、lm-eval（`accuracy_eval.py`）、
  样本与 `results/`。
- `docs/` —— `snapshot.md`（权威状态）、`adr/`、`findings/`、`design/`。

---

## 部署（快速上手）

`sglang_overlay/` 镜像了 sglang 的包结构；`scripts/deploy.sh` 把它 rsync 进目标机上
已安装的 sglang site-packages（不编译），随后照常启动 sglang 即可加载 RWKV-7：

```bash
# 通过环境变量配置目标（默认值仅为占位符）：
#   BOX = 目标机的 ssh 别名（本机安装用 "" 或 localhost）
#   SP  = 目标机上 sglang venv 的 site-packages 路径
BOX=<你的机器> SP=<site-packages 路径> bash scripts/deploy.sh
```

先用 BlinkDL `.pth` 经 `tools/convert_rwkv7_blinkdl_to_fla.py` 转成可加载权重，再用
sglang 正常起服务。量化推理加 `--quantization w8a8_int8`。

---

## 开发环境

- 远程机：1× RTX 3090，sglang **v0.5.10.post1**（torch 2.9.1/cu128）——之所以锁版本，
  是因为 sglang `main` 需要 CUDA 13，而开发机驱动仅支持 ≤12.9。
- 机器上无 GitHub/HF：参考代码在 Mac 上克隆到 `refs/`（已 gitignore）再 rsync 上去；
  模型走 ModelScope；密钥放在未纳入版本控制的 `~/.rwkv_secrets.sh`（**从不提交**）。
