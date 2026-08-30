# DFT Becke Partition 权重导数

这份文档将讨论 Becke partition 到二阶为止的权重导数。我们暂不讨论涉及到 Becke 配分的格点偏移 (grid-shift) 所对 DFT 一阶或二阶导数的贡献。

先前的讨论中，我们一般假定格点充分完备。这在不少情景里都是成立的 (杂化 GGA 频率与热力学分析、光谱计算等等)。但到 meta-GGA 任务时，动能密度的引入会导致比较严重的数值误差；这会放大格点不完备的影响 (进而导致数百 cm<sup>-1</sup> 的频率误差；而这在 GGA 下很少超过 5 cm<sup>-1</sup>)。

为了在 meta-GGA 下改善格点不完备导致的数值问题、或在 LDA/GGA 下一些特殊情景 (譬如有特殊精度要求的 AIMD 等)，我们有必要考虑格点偏移导致的导数贡献。

格点偏移的导数贡献一般有两部分：**格点权重** 与 **格点坐标**。其中，前者需要得到格点权重 $w_g$ 对原子坐标的导数。这份文档将着重于讨论该问题。REST 目前实现了 Becke partition，它也是计算化学程序中最常用的 partition 方法；我们仅局限于讨论该方法。

> **该文档包含借助 AI 推导与实现的算法**
>
> Becke Partition 二阶梯度的推导与实现中，一部分非常繁杂的部分由 AI 生成 (Claude Code + glm-5.2)。文档本身仍然是由人工撰写。
>
> 该文档本身也有 AI 参与；它是在与 AI 对话强讨论下完成的。第 1/2/4/5 节由人主导，第 3/6 节由 AI 主导。

> **Becke Partition 原始文献**
>
> A multicenter numerical integration scheme for polyatomic molecules.</br>
> Becke, A. D. *J. Chem. Phys.*, **1988**, *88* (4), 2547–2553. doi: [10.1063/1.454033](https://doi.org/10.1063/1.454033)

> **该文档采用 row-major 顺序**
>
> 在处理 Becke partition 时，我们不引入张量库。考虑到 Rust 在处理 vector 或 array 时，其风格是 row-major 顺序 (即最内层循环是最后一个指标)，因此我们在文档中也采用 row-major 顺序。

## 1. Becke partition 导数程序架构

**该架构设计由人与 AI 共同完成** (初始人工-AI 混合代码跑通的情况下，AI refactor 给出)。

> **实现决策：使用定长 array 的仿 SIMD 类型 `f64simd`**
>
> 在具体的程序实现里，我们会对格点维度 $g$ 作两重拆分：线程并行 task 拆分 (384-512)，以及线程内部串行 SIMD lane 拆分 (8)。
>
> 决策依据是
> - 在现代语言中，若仿 SIMD 类型实现了 +/-/*/div 等运算符重载，则程序实现与普通浮点类型 f64 差别较小。
> - Becke 格点实现细节有较复杂的代码；若不明确使用 SIMD lane 长度的数组，而是依每个浮点迭代循环，编译器不一定会作合理的自动向量化。
>
> 需要留意的部分
> - 该类型是仿 SIMD，并非真 SIMD。需要 `target-cpu=v3/v4/native` 等编译器选项才能发挥 SIMD 下的性能优势；但这也会导致不同微架构的 CPU 可能无法执行该二进制文件。
>
> 该设计模式在 PySCF 的 C 代码中也相当常见。

当前程序为平衡计算性能、内存复用、单个函数代码长度等考虑，框架上会稍显复杂。整个计算过程涉及到的类型分为
- **主线程级接口**
  - 预处理 (pub `BeckeMolTables`)，负责仅原子相关的计算 (原子间距离及其导数、Radii 矫正因子等)。这部分代码需要与格点分批处理过程完全解耦，避免依格点重复迭代循环计算或内存复制。
  - 用户参数输入 (pub `BeckePartitionArg`) 与输出 (pub `BeckePartitionOutput`)。输入参数决定要作多少阶导数、是否作 fused-contraction。
  - 计算设置 (`BeckePartitionContext`)，用于记录维度信息、导数阶数、输入向量数据的引用、输出结果的暂存等。
- **线程并行级接口**
  - 任务拆分 (`BatchTask`)，将全部格点拆分为互不重叠的区间 $[g_0, g_1)$。任务边界由格点归属方案 (`AtmIndices`) 决定：`ByGrid` 下按固定的 `nbatch` 均匀拆分 (任务内部可能混有不同原子的格点)；`ByAtom` 下任务不跨越原子的格点区间边界，每个任务只对应一个原子。
  - 线程缓冲 (`TaskBuffers`)，由并行迭代器的 per-worker 初始化分配一次、被该 worker 线程后续的所有任务复用。其中包括缩并部分和 (`TaskContraction`) 与 lane 暂存 (`LaneScratch`)。缩并部分和在每个任务结束时于互斥锁下 reduce 到输出缓冲。
- **格点迭代级接口**
  - lane 收集 (`gather_lane_batch`，输出 `LaneBatch`)，将任务的格点批次重新组织为定长 (`SIMDD` = 8) 的仿 SIMD lane。该函数是线程并行级与格点迭代级之间的桥接。
  - lane 求值与写回 (`process_lane` 所调度的 `eval_*`/`store_lane_*` 函数族，以及 `LaneScratch`、`LanePartition`、`LaneAttrib` 等)，以单个 lane (8 个格点) 为单位完成所有请求导数阶的求值、向输出缓冲的写回、以及向缩并部分和的累加。

三层之间的调用关系是：入口函数构建 `BeckeMolTables` 与 `BeckePartitionContext`，拆分出 `BatchTask` 后交给 rayon 并行迭代；每个任务先 gather 为 `LaneBatch`，再逐 lane 执行 `process_lane`。除主线程级接口的少数类型与入口函数外 (pub `BeckeMolTables`、`BeckePartitionArg`/`BeckePartitionOutput`、`AtmIndices` 与两个入口函数)，其余类型均为私有实现细节。

## 2. Becke partition 原始定义与程序实现

### 2.1 格点 Partition 概念

我们记 $A, B, M$ 为原子指标，$g$ 为格点指标。

| 变量名 | 变量公式 | 变量意义 | 维度 | 变量归属 |
|--|--|--|--|--|
| `weights`, `w` | $w_g$ | 格点权重 | $(g)$ | `LanePartition`</br>`BeckePartitionOutput` |
| `wquad`</br>`quadrature_weights` | $w_g^\text{quad}$ | 原始格点权重 | $(g)$ | input |
| `P` | $P_{Mg}$ | Becke 配分 | $(M, g)$ | `LaneScratch` |
| `Z` | $Z_M$ | Becke 配分求和 | $(M)$ | `LanePartition` |
| `Pg` | $\bar{P}_{g}$ | 格点对应原子的 Becke 配分 | $(g)$ | `LanePartition` |

格点权重记为 $w_g$；其中 $g$ 是格点指标，其对应的 3 分量 ($x, y, z$) 格点空间坐标是 $\bm{r}_g$。**格点权重 $w_g$ 是关于原子核坐标 $\bm{R}_A, \bm{R}_B, \ldots$ 的函数**。格点权重一般要满足下述性质：

$$
\sum_{g} w_g f(\bm{r}_g) \simeq \int d\bm{r} f(\bm{r}) \quad \forall f(\bm{r})
$$

格点权重 $w_g$ 从程序实现上，是原始格点 $w_g^\text{quad}$ 与 Becke 配分 $P_{Mg}$ 的组合：

$$
w_g = w_g^\text{quad} \frac{\bar{P}_{g}}{Z_g}
$$

- 原始格点是指，目前的 DFT 格点总是从每个原子中心产生球面格点，一般是 Lebedev 格点。这些 Lebedev 格点的权重即是 $w_g^\text{quad}$。每个原子中心的 Lebedev 格点 (及其权重) 理想情况下是完备的；但在实际计算中，这样的格点难以达到精确完备与性能的平衡。那么一种合理的做法是，先在每个原子中心生成完备的 Lebedev 格点，然后再通过 Becke 配分将这些格点组合成一个全局格点。这样就可以在保证精度的前提下，减少格点数量。
- **格点 $g$ 一定是某个原子 $M$ 所生成的 Lebedev 格点**。$\bar{P}_g$ 这里定义为 $\sum_{M} P_{M g} \delta_{g \in M}$，即格点 $g$ 对应的原子 $M$ 下的 Becke 配分数值。
- Becke 配分 $P_{Mg}$ 是指格点 $g$ 对应原子 $M$ 的 (非归一化的) 配分系数。通过 $Z_g = \sum_{M} P_{Mg}$ 进行归一化。

### 2.2 Becke partition 具体实现

| 变量名 | 变量公式 | 变量意义 | 维度 | 变量归属 |
|--|--|--|--|--|
| `adjustment_factor` | $a_{AB}$ | Radii 矫正因子 | $(A, B)$ | `BeckeMolTables` |
| `grid_coords`, `coords` | $\bm{r}_g$, $r_{g t}$ | 格点坐标 | $(g, 3)$ | input |
| `atm_coords` | $\bm{R}_A$, $R_{A t}$ | 原子核坐标 | $(A, 3)$ | `BeckeMolTables` |
| `dist` | $\Vert r \Vert_{Ag}$ | 原子核与格点距离 | $(A, g)$ | `LaneScratch` |
| `atm_dist` | $\Vert R \Vert_{AB}$ | 原子核间距离 | $(A, B)$ | `BeckeMolTables` |
| `mu` | $\mu_{A B g}$ | 椭球坐标差分 | | temporary |
| `s` | $s_{A B g}$ | switch 函数 | | temporary |
| `P` | $P_{Mg}$ | Becke 配分 | $(M, g)$ | `LaneScratch` |
| `Z` | $Z_M$ | Becke 配分求和 | $(M)$ | `LanePartition` |
| `Pg` | $\bar{P}_{g}$ | 格点对应原子的 Becke 配分 | $(g)$ | `LanePartition` |

Becke partition 的程序实现在 `eval_partition` 中；但其中一些重要的、但与格点无关的中间量，需要在进入 Becke partition 之前就计算好。这包括

- **Radii 矫正因子 `adjustment_factor` $a_{AB}$**。需要留意它通常是反对称矩阵。Radii 矫正的计算方法有不同的类型；REST (c.f. dftlibs/numgrid) 采用的是原始的 Becke 矫正策略，参见 Becke 1988 文章 eq (A3)-(A6)。这需要用户手动给出。
- **原子核间距离 `atm_dist` $\Vert R \Vert_{AB}$**。它是通过原子核坐标的二范数得到的 (`BeckeMolTables::new`, c.f. `fn dist3_naive`)：

  $$
  \Vert R \Vert_{AB} = 
  \begin{cases}
  \sqrt{\sum_t (R_{B t} - R_{A t})^2} & A \neq B \\
  \infty & A = B
  \end{cases}
  $$

在进入函数 `eval_partition` 后，需要进行的操作包括

- **原子核与格点距离 `dist` $\Vert r \Vert_{Ag}$**。它是通过原子核坐标与格点坐标的二范数得到的 (c.f. `fn dist3_hybrid`)：

  $$
  \Vert r \Vert_{Ag} = \sqrt{\sum_t (r_{g t} - R_{A t})^2}
  $$

- **椭球坐标差分 `mu` $\mu_{A B g}$**，该变量是中间量，随 $(A, B)$ 二重循环迭代：

  $$
  \mu_{A B g} = \frac{\Vert r \Vert_{Ag} - \Vert r \Vert_{Bg}}{\Vert R \Vert_{AB}}
  $$

- **switch 函数中间量 `f3` $f_3 (\mu)$** (c.f. `fn switch_f3`, `fn switch_f_hardness`)：

  $$
  \begin{align*}
  f_3 (\mu) &= p \circ p \circ p \circ \nu (\mu)
  \tag{eq. 20} \\
  p (\nu) &= \frac{3}{2} \nu - \frac{1}{2} \nu^3
  \tag{eq. 19} \\
  \nu (\mu) &= \mu + a (1 - \mu^2)
  \tag{eq. A2}
  \end{align*}
  $$

- **switch 函数 `s` $s_{A B g}$** 与 **Becke 配分 `P` $P_{Ag}$**。

  $$
  \begin{align*}
  s_{A B g} &= \frac{1}{2} (1 - f_3 (\mu_{A B g}))
  \tag{eq. 21} \\
  P_{A g} &= \prod_{B \neq A} s_{A B g}
  \tag{eq. 13}
  \end{align*}
  $$
  
  在实际程序实现中，switch 函数的结果会直接缩并入 Becke 配分中，不单独存储。同时，利用 $\text{antisymm}(A, B)$ 的特性，这里的计算量是可以减少一半的。因此，实际程序中的做法是，首先对 $P_{A g}$ 初始化为 1，且仅迭代 $0 = B < A = n_\mathrm{atom}$：

  $$
  \begin{align*}
  s_{A B g} &= \frac{1}{2} (1 - f_3 (\mu_{A B g})) \\
  s_{B A g} &= \frac{1}{2} (1 + f_3 (\mu_{A B g})) \\
  P_{A g} &\mathrel{\times}= s_{A B g} \\
  P_{B g} &\mathrel{\times}= s_{B A g}
  \end{align*}
  $$

- **Becke 配分求和 `Z` $Z_g$**。它是对 Becke 配分的归一化因子：

  $$
  Z_g = \sum_{M} P_{M g} \tag{eq. 22}
  $$

- **格点对应原子的 Becke 配分 `Pg` $\bar{P}_g$**。它是依据格点 $g$ 在生成 Lebedev 格点时的原子 $M$ 所对应的 Becke 配分 $P_{M g} \delta_{g \in M}$。在程序实际实现中，需要分为两种情况实现：
  - 格点所对应 Lebedev 格点的原子若是乱序的 (对应于 PySCF 的 `sort_grids=True` 的默认选项)，则走 `LaneAttrib::ByGrid` 的流程，以 `atm_idx` 的格点-原子映射表选取 $P_{M g}$ 的数值到 $\bar{P}_g$ 中。
  - 若一批 lane 中的所有格点都保证固定为一个原子 (对应于 PySCF 的 `sort_grids=False`，或现在 REST 的格点组织方式)，则走 `LaneAttrib::ByAtom` 的流程。这也是比较推荐的做法，这特别会在后续 DFT 导数中有较大帮助。

- **格点权重 `w` $w_g$**。它是 Becke partition 完成后的格点权重，也用于所有涉及 DFT 格点积分的计算任务里：

  $$
  w_g = w_g^\text{quad} \frac{\bar{P}_{g}}{Z_g} \tag{eq. 22}
  $$

## 3. Switch 函数 $f(\nu)$ 及其导数

我们先处理一个比较特殊的问题，Switch 函数 $f(\nu)$ 及其导数。

> **实现决策：对 $f_3(\nu)$ 实现的特化**
>
> 在原始公式中，$f_3(\nu) = p^n(\nu)$；这里的幂次是指函数复合，而非数值幂次。但几乎所有的程序都针对 $n = 3$ 的情况实现。因此，我们会作 $f_3$ 的特化 dispatch，而其他情况则作递归。

这部分的实现是比较机械的。必要时检查源代码即可。需要留意，这里的函数是 $f(\nu)$，以 $\nu$ 为自变量。逐项写出 ($p$ 的导数)：

$$
\begin{align*}
p' (\nu) &= \frac{3}{2} (1 - \nu^2) \\
p'' (\nu) &= -3 \nu
\end{align*}
$$

记 $f_0 = \nu$、$f_i = p (f_{i-1})$、$g_i = p' (f_{i-1})$。由链式法则，

$$
f_3' = g_2 g_1 g_0, \qquad f_3'' = -3 \big[ f_2 (g_1 g_0)^2 + f_1 g_2 g_0^2 + \nu g_2 g_1 \big]
$$

$f_3''$ 的形式来自对每一层复合应用乘积法则：每层贡献一个 $p'' (f_{i-1}) \, (\ldots)^2$ 型的项，而 $p'' = -3x$ 使系数恰为 $-3 f_{i-1}$。对任意 hardness，程序不展开该式，而是逐层递推 (初始值 $f_0 = \nu$、$f_0' = 1$、$f_0'' = 0$)：

$$
f_{k+1} = p (f_k), \qquad f_{k+1}' = p' (f_k)\, f_k', \qquad f_{k+1}'' = p'' (f_k)\, (f_k')^2 + p' (f_k)\, f_k''
$$

递推时需以旧值先算 $f_{k+1}''$、再算 $f_{k+1}'$ 与 $f_{k+1}$ (c.f. `switch_d2nu_f_hardness`)。

switch 函数本身是以 $\mu$ 为自变量的复合 $s = s_3 \circ \nu$ (eq. A1/A2)。$\nu$ 对 $\mu$ 的导数为

$$
\nu' = 1 - 2 a \mu, \qquad \nu'' = -2 a
$$

因此，$s$ 对 $\mu$ 的一阶、二阶导数为

$$
\begin{align*}
\frac{\partial s}{\partial \mu} &= -\frac{1}{2} f_3' \nu' \\
\frac{\partial^2 s}{\partial \mu^2} &= -\frac{1}{2} f_3'' (\nu')^2 - \frac{1}{2} f_3' \nu''
\end{align*}
$$

依 2.2 节的半迭代约定，同一个 $\mu_{ABg}$ 下 $s_{ABg} = \frac{1}{2} (1 - f_3)$ 与 $s_{BAg} = \frac{1}{2} (1 + f_3)$ 的各阶 $\mu$ 导数恰好相差一个符号。

后续推导 (5.1、6.1 节) 使用的是对数导数。以 $s_{\text{safe}}$ 替代 $s$ 作分母 (截断说明见 5.1 节)：

$$
\frac{\partial \log s}{\partial \mu} = \frac{1}{s_{\text{safe}}} \frac{\partial s}{\partial \mu}, \qquad \frac{\partial^2 \log s}{\partial \mu^2} = \frac{1}{s_{\text{safe}}} \frac{\partial^2 s}{\partial \mu^2} - \left( \frac{\partial \log s}{\partial \mu} \right)^2
$$

## 4. 梯度问题的平移不变性

我们先补充一个记号：格点 $g$ 总是某个原子所生成的 Lebedev 格点，记该生成原子为 $A_g$，并以 $g \in A_g$ 表示“$g$ 是原子 $A_g$ 所生成的格点”。

我们在后续的推导中，会将格点坐标 $\bm{r}_g$ 视为不依赖于任何原子核坐标 $\bm{R}_A$ 的独立变量。**从程序角度来说，这是错误的**。在实际程序中，当我们从一个坐标被微扰的原子出发重新构建格点时，格点坐标会随原子坐标的变化而变化：格点 $g \in A_g$ 随其生成原子 $A_g$ 作刚性平移。因此，原则上当然要考虑格点坐标对原子坐标的导数；若忽略它，许多中间量的解析导数与数值导数是对不上的。

但引入格点坐标对原子坐标的依赖后，推导与计算会变得复杂。我们的处理策略是：**从现在开始忘记 $\bm{r}_g$ 对 $\bm{R}_{A_g}$ 的依赖，直到最后一步结算 $w_g$ 的导数时，再以本节的方法统一修正**。修正依靠的是格点权重在整体平移下的不变性；经过这一处理，最终结果与数值导数一致，尽管中间量无法与数值导数核验。

依赖结构值得分类讨论。由于 $\bm{r}_g$ 仅依赖于 $\bm{R}_{A_g}$，而不依赖于其他原子坐标：

- 对于 $A \neq A_g$：$\bm{r}_g$ 不随 $\bm{R}_A$ 变化，按照 $\bm{r}_g$ 独立的方式计算的偏导数就是真实导数，结果正确。
- 对于 $A = A_g$：按照 $\bm{r}_g$ 独立的方式计算的偏导数是错误的；其真实导数是全导数，可由其余原子的正确偏导数反推：

$$
\frac{\mathrm{d} w_g}{\mathrm{d} R_{A_g t}} = - \sum_{M \neq A_g} \frac{\partial w_g}{\partial R_{M t}} \quad (g \in A_g)
$$

上式左端是全导数，右端的偏导数均在“$\bm{r}_g$ 独立”的约定下计算。下面对上式作推导。

首先，我们将原子核坐标与格点坐标同时作为独立变量对待，写为

$$
w_g = w_g (\bm{r}_g, \bm{R}_A, \bm{R}_B, \ldots)
$$

格点权重是平移不变的：对所有坐标施加同一个平移量 $\bm{t}$，格点权重不变，

$$
w_g (\bm{r}_g + \bm{t}, \bm{R}_A + \bm{t}, \bm{R}_B + \bm{t}, \ldots) = w_g (\bm{r}_g, \bm{R}_A, \bm{R}_B, \ldots)
$$

在 $\bm{t} = \bm{0}$ 处对上式作 $\bm{t}$ 的全导数，得到

$$
\frac{\partial w_g}{\partial \bm{r}_g} + \sum_M \frac{\partial w_g}{\partial \bm{R}_M} = 0
$$

现在考虑 $g \in A_g$，并求 $w_g$ 对生成原子坐标 $\bm{R}_{A_g}$ 的导数。Lebedev 角向格点与径向格点都不随原子核坐标旋转 (格点定义没有取向依赖)，$\bm{R}_{A_g}$ 平移 $\bm{t}$ 后 $\bm{r}_g$ 也平移 $\bm{t}$；用数学的方式表述 (下述单位矩阵为 3×3)，

$$
\frac{\mathrm{d} \bm{r}_g}{\mathrm{d} \bm{R}_{A_g}} = \mathbf{I}
$$

于是 $w_g$ 对 $\bm{R}_{A_g}$ 的全导数为

$$
\frac{\mathrm{d} w_g}{\mathrm{d} \bm{R}_{A_g}} = \frac{\partial w_g}{\partial \bm{R}_{A_g}} + \frac{\partial w_g}{\partial \bm{r}_g} \mathbf{I} = \frac{\partial w_g}{\partial \bm{R}_{A_g}} + \frac{\partial w_g}{\partial \bm{r}_g}
$$

将平移不变性给出的 $\frac{\partial w_g}{\partial \bm{r}_g} = - \sum_M \frac{\partial w_g}{\partial \bm{R}_M}$ 代入，即得

$$
\frac{\mathrm{d} w_g}{\mathrm{d} \bm{R}_{A_g}} = \frac{\partial w_g}{\partial \bm{R}_{A_g}} - \sum_M \frac{\partial w_g}{\partial \bm{R}_M} = - \sum_{M \neq A_g} \frac{\partial w_g}{\partial \bm{R}_M}
$$

取其 $t$ 分量即是前述修正公式，推导完成。

回顾整个论证，我们只用到两件事：$w_g$ 的显式变量集合是 $(\bm{r}_g; \bm{R}_A, \bm{R}_B, \ldots)$，以及 $w_g$ 在整体平移下不变。论证并不依赖 Becke partition 的具体函数形式，因此对其他格点配分方案、乃至更一般的“随格点携带的量”的梯度计算同样成立。格点 $g$ 的生成原子是唯一必须作全导数处理的原子，其余原子的偏导数本来就正确；因此程序实现上可以先用“$\bm{r}_g$ 独立”的约定完成所有偏导数的计算，最后以修正公式反推生成原子的全导数。

> **补充说明：二阶导数的平移不变性修正**
>
> 二阶导数 $\partial^2 w_g / \partial R_{A t} \partial R_{B s}$ 的修正遵循同一原理；但由于 $A$、$B$ 两个导数指标各自都可能落在生成原子上，需要区分行 ($A = A_g$)、列 ($B = A_g$) 与角点 ($A = B = A_g$) 三类求和式。我们推迟到讨论二阶梯度的章节再展开。

## 5. Becke partition 一阶梯度程序实现

一阶梯度在程序上分两步实现。第一步 (`eval_switch_pair_pass` 中的一阶梯度部分，5.2 节) 在零阶量 ($P_{M g}$, $\Vert r \Vert_{A g}$, $Z_g$, $\bar P_g$ 等，见 2.2 节) 的基础上，逐原子对累加两个 3-dim 中间导数量 $\frac{\partial Z_g}{\partial R_{At}}$ 与 $\frac{\partial \bar P_g}{\partial R_{At}}$。第二步 (`eval_lane_dw`，5.3 节) 对 $w_g = w_g^{\text{quad}} \bar P_g / Z_g$ 作商法则结算，并作 4 节的平移不变性修正。

其中，连乘积 $P_{Mg}$ 的导数是推导里第一个非平凡的问题；我们先以 5.1 节单独说明。

### 5.1 连乘积 $P_{Mg}$ 的导数

对连乘积直接使用乘积法则，会产生 $O(n_\mathrm{atom})$ 项、每项含 $n_\mathrm{atom} - 1$ 个因子的展开式；惯用的处理方式是对连乘积取对数，将连乘转为求和。对 eq. 13 两边取对数，

$$
\log P_{Mg} = \sum_{N \neq M} \log s_{MNg}
$$

再作微分，

$$
\frac{\partial P_{Mg}}{P_{Mg}} = \partial \log P_{Mg} = \sum_{N \neq M} \partial \log s_{MNg}
$$

整理得到连乘积导数的基本公式：

$$
\partial P_{Mg} = P_{Mg} \sum_{N \neq M} \partial \log s_{MNg}
$$

这个形式有两个好处。

- **数值稳定性**：$P_{Mg}$ 与其导数都以乘法因子出现，$P_{Mg}$ 从不进入分母。Becke 配分在远离原子 $M$ 的格点处可以非常小，对 $P_{Mg}$ 作除法会放大相对误差。相形之下，$\partial \log s$ 中出现的 $1 / s$ 是有界的：程序对过小的 $s$ 作下限截断 ($s_{\text{safe}} = \max(s, 10^{-14})$，c.f. `INVTOL`)。
- **缩并结构**：后续 (5.3 节) 真正需要的不是 4-dim 张量 $\frac{\partial P_{Mg}}{\partial R_{At}}$，而是它的两个缩并 $\frac{\partial Z_g}{\partial R_{At}} = \sum_M \frac{\partial P_{Mg}}{\partial R_{At}}$ 与 $\frac{\partial \bar P_g}{\partial R_{At}} = \frac{\partial P_{A_g g}}{\partial R_{At}}$，均为 3-dim 张量 $(A, t, g)$。利用上式可以逐原子对累加，不必显式构造任何 $(M, A, t, g)$ 中间张量。

另一个关键的结构事实是，$s$ 是 $\mu$ 的一元函数，因此张量 $s_{MNg}$ 只对与自己配对的 $\mu_{MNg}$ 有导数：

$$
\frac{\partial s_{MNg}}{\partial \mu_{ABg}} = \frac{\partial s_{MNg}}{\partial \mu_{MNg}} \delta_{AM} \delta_{BN}
$$

结合链式法则，$P_{Mg}$ 对原子坐标的导数为

$$
\frac{\partial P_{Mg}}{\partial R_{At}} = P_{Mg} \sum_{N \neq M} \frac{\partial \log s_{MNg}}{\partial \mu_{MNg}} \frac{\partial \mu_{MNg}}{\partial R_{At}}
$$

其中 switch 函数对数导数 $\frac{\partial \log s}{\partial \mu}$ 的显式表达式由 3 节给出，这里不再展开。

### 5.2 一阶梯度重要中间量计算

| 变量名 | 变量公式 | 变量意义 | 维度 | 变量归属 |
|--|--|--|--|--|
| `dR_dist` | $\Vert \partial r \Vert_{Atg}$ | 格点距离对原子坐标导数 | $(A, t, g)$ | `LaneScratch` |
| `dR_atm_dist` | $\Vert \partial R \Vert_{ABt}$ | 原子间距对原子坐标导数 | $(A, B, t)$ | `BeckeMolTables` |
| `dR_mu_roleA` | $\partial \mu_{ABg} / \partial R_{At}$</br>(role A) | 椭球坐标</br>role A 导数 | | temporary |
| `dR_mu_roleB` | $\partial \mu_{ABg} / \partial R_{Bt}$</br>(role B) | 椭球坐标</br>role B 导数 | | temporary |
| `dmu_log_sA` | $\partial \log s_{ABg} / \partial \mu_{ABg}$ | switch 对数导数 | | temporary |
| `dmu_log_sB` | $\partial \log s_{BAg} / \partial \mu_{ABg}$ | switch 对数导数 (反对侧) | | temporary |
| `dR_Z` | $\partial Z_g / \partial R_{At}$ | 归一化因子一阶导数 | $(A, t, g)$ | `LaneScratch` |
| `dR_Pg` | $\partial \bar P_g / \partial R_{At}$ | 格点配分分子一阶导数 | $(A, t, g)$ | `LaneScratch` |

本小节对应 `eval_switch_pair_pass` 的一阶梯度部分。需要进行的操作包括

- **格点距离导数 `dR_dist` $\Vert \partial r \Vert_{Atg}$**。它是对 $R_{At}$ 的导数，几何上是格点指向原子 $A$ 的单位向量的 $t$ 分量：

  $$
  \Vert \partial r \Vert_{Atg} := \frac{\partial \Vert r \Vert_{Ag}}{\partial R_{At}} = \frac{R_{At} - r_{gt}}{\Vert r \Vert_{Ag}}
  $$

- **原子间距导数 `dR_atm_dist` $\Vert \partial R \Vert_{ABt}$**。它与格点无关，在 `BeckeMolTables` 中预计算 (deriv $\geqslant 1$ 时)：

  $$
  \Vert \partial R \Vert_{ABt} := \frac{\partial \Vert R \Vert_{AB}}{\partial R_{At}} = \frac{R_{At} - R_{Bt}}{\Vert R \Vert_{AB}}
  $$

  需要留意其关于 $(A, B)$ 是反对称的：$\Vert \partial R \Vert_{ABt} = - \Vert \partial R \Vert_{BAt}$，即对右侧原子求导时变号。

- **椭球坐标的 role 导数 `dR_mu_roleA`/`dR_mu_roleB`**。真实的导数张量 $\frac{\partial \mu_{MNg}}{\partial R_{At}}$ 是 5-dim 的 $(A, M, N, t, g)$，但其中包含 Kronecker delta：

  $$
  \frac{\partial \mu_{MNg}}{\partial R_{At}} = \frac{\partial \mu_{ANg}}{\partial R_{At}} \delta_{AM} + \frac{\partial \mu_{MAg}}{\partial R_{At}} \delta_{AN}
  $$

  实际计算将 delta 提前缩并：导数原子要么是 $\mu$ 的左端点 (role A)、要么是右端点 (role B)，每对 $(A, B)$ 只需两个 3-分量临时量：

  $$
  \begin{align*}
  \frac{\partial \mu_{ABg}}{\partial R_{At}} \; (\text{role A}) &= \frac{1}{\Vert R \Vert_{AB}} \big( \Vert \partial r \Vert_{Atg} - \mu_{ABg} \Vert \partial R \Vert_{ABt} \big) \\
  \frac{\partial \mu_{ABg}}{\partial R_{Bt}} \; (\text{role B}) &= \frac{1}{\Vert R \Vert_{AB}} \big( -\Vert \partial r \Vert_{Btg} + \mu_{ABg} \Vert \partial R \Vert_{ABt} \big)
  \end{align*}
  $$

- **逐原子对累加 `dR_Z`/`dR_Pg`**。沿用 2.2 节的半迭代约定 $0 \leqslant B < A$：$s_{ABg}$ 与 $s_{BAg} = 1 - s_{ABg}$ 都视为 $\mu_{ABg}$ 的函数，由同一次 $f_3 (\mu_{ABg})$ 求值给出；相应地，两个对数导数都取对 $\mu_{ABg}$ 的导数 (注意 $\partial \log s_{BAg} / \partial \mu_{ABg}$ 与 5.1 节中按 $\mu_{BAg}$ 定义的量相差一个符号)。每个原子对 $(A, B)$ 对两个累加器的贡献是

  $$
  \begin{align*}
  \frac{\partial Z_g}{\partial R_{At}} &\mathrel{+}= \Big( P_{Ag} \frac{\partial \log s_{ABg}}{\partial \mu_{ABg}} + P_{Bg} \frac{\partial \log s_{BAg}}{\partial \mu_{ABg}} \Big) \frac{\partial \mu_{ABg}}{\partial R_{At}} \quad (\text{role A}) \\
  \frac{\partial \bar P_g}{\partial R_{At}} &\mathrel{+}= \Big( \delta_{A A_g} P_{Ag} \frac{\partial \log s_{ABg}}{\partial \mu_{ABg}} + \delta_{B A_g} P_{Bg} \frac{\partial \log s_{BAg}}{\partial \mu_{ABg}} \Big) \frac{\partial \mu_{ABg}}{\partial R_{At}} \quad (\text{role A})
  \end{align*}
  $$

  role B 的两式同形，将 $(A, t)$ 替换为 $(B, t)$、链式因子换用 role B 导数即可 (圆括号系数不变)。两式中的系数对应程序变量 `common_Z` 与 `common_Pg`；$\delta_{A A_g}$、$\delta_{B A_g}$ 表明分子只接收生成原子 $A_g$ 的 $P$ 的贡献，程序上由 `LaneAttrib` 的选取实现 (`ByGrid` 逐 lane 掩码、`ByAtom` 确定原子)。

- **两遍结构**。连乘积 $P_{M g}$ 与距离 $\Vert r \Vert_{A g}$ 必须完整生成后才能参与导数计算 (5.1 节的公式以完整的 $P_{Mg}$ 为因子)，因此一阶梯度是继 `eval_partition` 之后的第二遍 pair 循环，两者不能合并。一阶累加器 `dR_Z`/`dR_Pg` 在每个 lane 开始时清零。

### 5.3 权重一阶梯度

本小节对应 `eval_lane_dw`，输出格点权重一阶导数 `dw` $\frac{\partial w_g}{\partial R_{At}}$，维度 $(A, t, g)$。

- **商法则结算**。对 $w_g = w_g^{\text{quad}} \bar P_g / Z_g$ 与 5.2 节给出的两个中间量作商法则：

  $$
  \frac{\partial w_g}{\partial R_{At}} = w_g^{\text{quad}} \left( \frac{1}{Z_g} \frac{\partial \bar P_g}{\partial R_{At}} - \frac{\bar P_g}{Z_g^2} \frac{\partial Z_g}{\partial R_{At}} \right)
  $$

- **平移不变性修正**。上式是在“$\bm{r}_g$ 独立”约定下的偏导数；依第 4 节的讨论，格点 $g$ 在生成原子 $A$ 的导数需要依平移不变性的推论，从全导数的结果作替换：

  $$
  \frac{\partial w_g}{\partial R_{A_g t}} = - \sum_{M \neq A_g} \frac{\partial w_g}{\partial R_{M t}}
  $$

  程序分两步实现：先累计负的行和 $\texttt{dw\_neg\_sum}_t = - \sum_M \frac{\partial w_g}{\partial R_{Mt}}$，再将其回填到生成原子的行 (由于其余行均正确，回填等价于 `dw[A_g, t] += dw_neg_sum[t]`)。回填方式依 `AtmIndices` 而定：`ByAtom` 下生成原子对整个 lane 的浮点数确定，作整寄存器的行更新；`ByGrid` 下则逐浮点数更新。

## 6. Becke partition 二阶梯度程序实现

二阶梯度与一阶 (5 节) 共用同样的输入量与 pair 循环框架，小节组织也与 5 节平行：6.1 给出连乘积二阶导数的数学准备，6.2 对应 `eval_switch_pair_pass` 的二阶部分 (逐对累加)，6.3 对应 `eval_lane_ddw` (结算)。但二阶并非一阶的简单重复：rank-1 交叉项的出现改变了 pair 循环与结算的分工 (见 6.1 末尾)。程序上，仅当请求二阶输出 (`ddw` 或其缩并 `ddc`) 时才运行二阶机制 (`do_deriv2`)。

### 6.1 连乘积 $P_{Mg}$ 的二阶导数

5.1 的论证模式可以直接推进到二阶。对 $\log P_{Mg} = \sum_{N \neq M} \log s_{MNg}$ 再作一次微分：

$$
\frac{\partial^2 \log P_{Mg}}{\partial R_{At} \partial R_{Bs}} = \sum_{N \neq M} \frac{\partial^2 \log s_{MNg}}{\partial R_{At} \partial R_{Bs}}
$$

每个 switch 因子的二阶导数由链式法则给出 (对数导数的显式表达式见 3 节)：

$$
\frac{\partial^2 \log s_{MNg}}{\partial R_{At} \partial R_{Bs}} = \frac{\partial^2 \log s_{MNg}}{\partial \mu_{MNg}^2} \frac{\partial \mu_{MNg}}{\partial R_{At}} \frac{\partial \mu_{MNg}}{\partial R_{Bs}} + \frac{\partial \log s_{MNg}}{\partial \mu_{MNg}} \frac{\partial^2 \mu_{MNg}}{\partial R_{At} \partial R_{Bs}}
$$

对 5.1 的基本公式 $\partial P_{Mg} = P_{Mg}\, \partial \log P_{Mg}$ 再作一次微分 (乘积法则)，得到二阶的基本公式：

$$
\frac{\partial^2 P_{Mg}}{\partial R_{At} \partial R_{Bs}} = P_{Mg} \left( \frac{\partial^2 \log P_{Mg}}{\partial R_{At} \partial R_{Bs}} + \frac{\partial \log P_{Mg}}{\partial R_{At}} \frac{\partial \log P_{Mg}}{\partial R_{Bs}} \right)
$$

5.1 所述的两个好处在二阶同样成立：$P_{Mg}$ 仍只以乘法因子出现；逐对结构仍允许直接缩并。但二阶多出一个**结构性变化**：

- **rank-1 交叉项是跨对的**。$\partial^2 \log P_{Mg}$ 逐对可加 (每个 $N$ 单独贡献，无交叉)；但第二项 $\partial \log P_{Mg} \otimes \partial \log P_{Mg}$ 是同一个求和的平方，展开后含**不同原子对** $(N \neq N')$ 因子的乘积。它不能并入逐对的 pair 循环，必须在 $\partial \log P_{Mg}$ 完整生成之后、于结算阶段统一补上。

这决定了二阶与一阶不同的程序分工：pair 循环 (6.2 节) 存储对数一阶导数并累加逐对部分；结算 (6.3 节) 补上交叉项后再作商法则。相应地，一阶 (5.2 节) 不需要存储 $\partial \log P_{Mg}$ (缩并后即丢弃)，二阶则必须存储它 (c.f. `dR_log_P`)——这也是二阶 lane 暂存的内存开销达到 $O(n_\mathrm{atm}^2)$ 的主要来源。

### 6.2 二阶梯度重要中间量计算

| 变量名 | 变量公式 | 变量意义 | 维度 | 变量归属 |
|--|--|--|--|--|
| `PrM` | $\mathrm{Proj} (\bm r_{M}) / \Vert r \Vert_{Mg}$ | 格点方向单位向量的投影矩阵 | $(M, t, s)$ | `LaneScratch` |
| `dR_log_P` | $\partial \log P_{Mg} / \partial R_{Ct}$ | 连乘积对数一阶导数 | $(M, C, t)$ | `LaneScratch` |
| `ddmu_log_sA` | $\partial^2 \log s_{ABg} / \partial \mu_{ABg}^2$ | switch 二阶对数导数 | | temporary |
| `ddmu_log_sB` | $\partial^2 \log s_{BAg} / \partial \mu_{ABg}^2$ | switch 二阶对数导数 (反对侧) | | temporary |
| `ddR_mu_roleAA` | $\partial^2 \mu_{ABg} / \partial R_{At} \partial R_{As}$ (role AA) | 椭球坐标二阶 role 导数 | | temporary |
| `ddR_mu_roleAB` | $\partial^2 \mu_{ABg} / \partial R_{At} \partial R_{Bs}$ (role AB) | 同上；role BA 为其 $(t, s)$ 转置 | | temporary |
| `ddR_mu_roleBB` | $\partial^2 \mu_{ABg} / \partial R_{Bt} \partial R_{Bs}$ (role BB) | 同上 | | temporary |
| `ddR_Z` | $\partial^2 Z_g / \partial R_{At} \partial R_{Bs}$ | 归一化因子二阶导数 (截至 pair 循环，不含交叉项) | $(A, t, B, s)$ | `LaneScratch` |
| `ddR_Pg` | $\partial^2 \bar P_g / \partial R_{At} \partial R_{Bs}$ | 格点配分分子二阶导数 (截至 pair 循环，不含交叉项) | $(A, t, B, s)$ | `LaneScratch` |

本小节对应 `eval_switch_pair_pass` 的二阶部分；一阶复用量 (`dR_dist`、`dmu_log_sA/B`、role 一阶导数等) 见 5.2 节表格。二阶的逐对累加与 5.2 的一阶累加合并在同一个 pair 循环中：它们遍历相同的 $(A, B)$ 对，且 deriv 2 下 switch 求值一次给出 $f_3, f_3', f_3''$。需要进行的操作包括

- **椭球坐标的二阶 role 导数 `ddR_mu_roleAA/AB/BB`**。记 $f = \Vert r \Vert_{Ag} - \Vert r \Vert_{Bg}$、$g = \Vert R \Vert_{AB}$ ($\mu = f / g$)，并将 5.2 节的分量式一阶导数改写为向量形式 (单位向量 $\bm r_A = (\bm R_A - \bm r_g) / \Vert r \Vert_{Ag}$ 的分量即 $\Vert \partial r \Vert_{Atg}$、$\bm R_{AB}$ 同理)：

  $$
  \frac{\partial \mu}{\partial \bm R_A} = \frac{\bm r_A - \mu \bm R_{AB}}{\Vert R \Vert_{AB}}, \qquad \frac{\partial \mu}{\partial \bm R_B} = \frac{-\bm r_B + \mu \bm R_{AB}}{\Vert R \Vert_{AB}}
  $$

  二阶导数由商法则通式给出：

  $$
  \frac{\partial^2 (f / g)}{\partial X \partial Y} = \frac{f_{XY} g - (f_X g_Y + g_X f_Y) - f g_{XY}}{g^2} + \frac{2 f\, g_X g_Y}{g^3}
  $$

  两个求导方向 $X, Y$ 各自落在 $\mu$ 的左端点 ($\bm R_A$) 或右端点 ($\bm R_B$) 上，共四种 role 组合；各 role 的 $(f_X, f_Y, f_{XY}, g_X, g_Y, g_{XY})$ 取值如下 (记 $\mathrm{Proj} (\bm v) = \mathbf{I} - \bm v \bm v^{\mathsf T}$)：

  | role | $f_X$ | $f_Y$ | $f_{XY}$ | $g_X$ | $g_Y$ | $g_{XY}$ |
  |---|---|---|---|---|---|---|
  | AA | $\bm r_A$ | $\bm r_A$ | $\mathrm{Proj} (\bm r_A) / \Vert r \Vert_{Ag}$ | $\bm R_{AB}$ | $\bm R_{AB}$ | $\mathrm{Proj} (\bm R_{AB}) / \Vert R \Vert_{AB}$ |
  | AB | $\bm r_A$ | $-\bm r_B$ | $\bm 0$ | $\bm R_{AB}$ | $-\bm R_{AB}$ | $-\mathrm{Proj} (\bm R_{AB}) / \Vert R \Vert_{AB}$ |
  | BB | $-\bm r_B$ | $-\bm r_B$ | $-\mathrm{Proj} (\bm r_B) / \Vert r \Vert_{Bg}$ | $-\bm R_{AB}$ | $-\bm R_{AB}$ | $\mathrm{Proj} (\bm R_{AB}) / \Vert R \Vert_{AB}$ |

  其中 $\mathrm{Proj} (\bm r_M) / \Vert r \Vert_{Mg}$ 只依赖原子 $M$，每 lane 预计算一次 (`PrM`)；$\mathrm{Proj} (\bm R_{AB}) / \Vert R \Vert_{AB}$ 依对计算。role BA 与 role AB 在 $(t, s)$ 上互为转置，不单独计算。程序实现为对上表的直接代入 (c.f. `eval_switch_pair_pass` 中的 `d2mu`)。

- **对数一阶导数的存储 `dR_log_P`**。与一阶 (5.2 节) 不同，$\partial \log P_{Mg} / \partial R_{Ct}$ 不再缩并后丢弃，而需要为 6.3 节的交叉项完整存储。每个原子对 $(A, B)$ 的写入为

  $$
  \frac{\partial \log P_{Ag}}{\partial R_{Ct}} \mathrel{+}= \frac{\partial \log s_{ABg}}{\partial \mu_{ABg}} \frac{\partial \mu_{ABg}}{\partial R_{Ct}}, \qquad \frac{\partial \log P_{Bg}}{\partial R_{Ct}} \mathrel{+}= \frac{\partial \log s_{BAg}}{\partial \mu_{ABg}} \frac{\partial \mu_{ABg}}{\partial R_{Ct}} \qquad (C \in \{A, B\})
  $$

- **逐对 L2 累加 `ddR_Z`/`ddR_Pg`**。按 6.1 节的基本公式，$\partial^2 P_{Mg}$ 中逐对可加的部分是 $P_{Mg}\, \partial^2 \log P_{Mg}$；每个原子对 $(A, B)$ 对两个累加器的 (role AA 块) 贡献为

  $$
  \begin{align*}
  \frac{\partial^2 Z_g}{\partial R_{At} \partial R_{As}} &\mathrel{+}= \Big( P_{Ag} \frac{\partial^2 \log s_{ABg}}{\partial \mu_{ABg}^2} + P_{Bg} \frac{\partial^2 \log s_{BAg}}{\partial \mu_{ABg}^2} \Big) \frac{\partial \mu_{ABg}}{\partial R_{At}} \frac{\partial \mu_{ABg}}{\partial R_{As}} \\
  &\quad + \Big( P_{Ag} \frac{\partial \log s_{ABg}}{\partial \mu_{ABg}} + P_{Bg} \frac{\partial \log s_{BAg}}{\partial \mu_{ABg}} \Big) \frac{\partial^2 \mu_{ABg}}{\partial R_{At} \partial R_{As}}
  \end{align*}
  $$

  $$
  \begin{align*}
  \frac{\partial^2 \bar P_g}{\partial R_{At} \partial R_{As}} &\mathrel{+}= \Big( \delta_{A A_g} P_{Ag} \frac{\partial^2 \log s_{ABg}}{\partial \mu_{ABg}^2} + \delta_{B A_g} P_{Bg} \frac{\partial^2 \log s_{BAg}}{\partial \mu_{ABg}^2} \Big) \frac{\partial \mu_{ABg}}{\partial R_{At}} \frac{\partial \mu_{ABg}}{\partial R_{As}} \\
  &\quad + \Big( \delta_{A A_g} P_{Ag} \frac{\partial \log s_{ABg}}{\partial \mu_{ABg}} + \delta_{B A_g} P_{Bg} \frac{\partial \log s_{BAg}}{\partial \mu_{ABg}} \Big) \frac{\partial^2 \mu_{ABg}}{\partial R_{At} \partial R_{As}}
  \end{align*}
  $$

  其余三个 role 块 (AB、BA、BB) 同形：将两个链式因子与 $\partial^2 \mu$ 换用对应 role 的量即可，圆括号系数不变。两式的系数对应程序变量 `common_Z`/`common_dd` 与 `c1_Pg`/`cdd_Pg`；$\delta$ 选取与 5.2 节相同，由 `LaneAttrib` 实现。

- **零初始化纪律**。上述累加中，对角块 ($M$ 行的 $C = M$ 列、`ddR_Z`/`ddR_Pg` 的 $(A, A)$、$(B, B)$ 块) 接收多个对的贡献、每 lane 清零后累加；非对角块只有唯一原子对的贡献、直接赋值。因此每 lane 只需清零 $O(n_\mathrm{atm})$ 个对角元，$O(n_\mathrm{atm}^2)$ 的非对角块由唯一写入者覆盖，不需要整体清零。

### 6.3 权重二阶梯度

本小节对应 `eval_lane_ddw`，输出格点权重二阶导数 `ddw` $\frac{\partial^2 w_g}{\partial R_{At} \partial R_{Bs}}$，维度 $(A, t, B, s)$ (row-major `[natm, 3, natm, 3, ngrids]`)。

- **交叉项补全 (rank-1 矩阵)**。6.2 节结束时，`ddR_Z`/`ddR_Pg` 已包含 $\partial^2 \log P$ 的全部逐对内容；尚缺的是 6.1 节末尾指出的 rank-1 项。对 $Z_g$，它是矩阵

  $$
  C_{At, Bs} = \sum_M P_{Mg} \frac{\partial \log P_{Mg}}{\partial R_{At}} \frac{\partial \log P_{Mg}}{\partial R_{Bs}}
  $$

  (对称：$C_{At, Bs} = C_{Bs, At}$)。对 $\bar P_g$， rank-1 项只有 $M = A_g$ 一项，即 $\bar P_g \frac{\partial \log P_{A_g g}}{\partial R_{At}} \frac{\partial \log P_{A_g g}}{\partial R_{Bs}}$ (程序收集 `dR_log_P` 的 $A_g$ 行为 `dlog_Ag`)。补全后即为完整的 $\frac{\partial^2 Z_g}{\partial R_{At} \partial R_{Bs}}$ 与 $\frac{\partial^2 \bar P_g}{\partial R_{At} \partial R_{Bs}}$。

  > **实现决策：交叉项 rank-1 矩阵的分块计算**
  >
  > $C$ 的每个输出元需要对 $M$ 求和，而 `dR_log_P` 的总量是 $(3 n_\mathrm{atm})^2$ 个 `f64simd` 寄存器 ($n_\mathrm{atm} = 10$ 时约 58 KB)，超出 L1 cache。程序按 $12 \times 12$ 个 `f64simd` 的 tile 分块计算 (`CROSS_CB`)：每个 tile 的累加器约 $12 \times 12 \times 8 \times 8\,\mathrm{B} \approx 9$ KB，保持 L1 常驻；`dR_log_P` 的行改为流式读取，被重读的次数从每个输出行一次降为每 $12$ 行一次。利用 $C$ 的对称性，只计算上三角 tile 并镜像存储。

- **商法则结算**。记 $q_g = \bar P_g / Z_g$ (**仅本小节使用**；$w_g = w_g^{\text{quad}} q_g$)，商法则给出

  $$
  \frac{\partial^2 w_g}{\partial R_{At} \partial R_{Bs}} = w_g^{\text{quad}} \left\{ \frac{1}{Z_g} \left[ \frac{\partial^2 \bar P_g}{\partial R_{At} \partial R_{Bs}} - \frac{\partial q_g}{\partial R_{Bs}} \frac{\partial Z_g}{\partial R_{At}} - q_g \frac{\partial^2 Z_g}{\partial R_{At} \partial R_{Bs}} \right] - \frac{1}{Z_g} \frac{\partial q_g}{\partial R_{At}} \frac{\partial Z_g}{\partial R_{Bs}} \right\}
  $$

  其中 $\frac{\partial q_g}{\partial R_{At}} = \frac{1}{Z_g} \frac{\partial \bar P_g}{\partial R_{At}} - \frac{\bar P_g}{Z_g^2} \frac{\partial Z_g}{\partial R_{At}}$，即 5.3 节商法则的括号内除以 $w_g^{\text{quad}}$。

- **平移不变性修正**。上式是“$\bm r_g$ 独立”约定下的偏导数 (`ddw_partial`)；依 4 节及其末尾的补充说明，行、列与角点需要替换为全导数：

  $$
  \frac{\partial^2 w_g}{\partial R_{A_g t} \partial R_{Bs}} = - \sum_{M \neq A_g} \frac{\partial^2 w_g}{\partial R_{Mt} \partial R_{Bs}}, \qquad
  \frac{\partial^2 w_g}{\partial R_{At} \partial R_{A_g s}} = - \sum_{N \neq A_g} \frac{\partial^2 w_g}{\partial R_{At} \partial R_{Ns}}
  $$

  $$
  \frac{\partial^2 w_g}{\partial R_{A_g t} \partial R_{A_g s}} = \sum_{M \neq A_g} \sum_{N \neq A_g} \frac{\partial^2 w_g}{\partial R_{Mt} \partial R_{Ns}}
  $$

  程序在商法则的逐元素扫描中顺带累计三个和：`fullA`$[B, t, s] = \sum_A$、`fullB`$[A, t, s] = \sum_B$ 与 `fullAB`$[t, s] = \sum_{A, B}$；行修正即 $-\,$`fullA`$[B, t, s] + \texttt{ddw\_partial}[A_g t, B s]$，列修正即 $-\,$`fullB`$[A, t, s] + \texttt{ddw\_partial}[A t, A_g s]$，角点为 `fullAB`$[t, s] - \,$`fullA`$[A_g] - \,$`fullB`$[A_g] + \texttt{ddw\_partial}[A_g t, A_g s]$ (未修正部分值只涉及正确的行列，可直接复用)。`ByAtom`/`ByGrid` 的差异同 5.3 节。

- **验证**。其一，Hessian 对称性 $\frac{\partial^2 w_g}{\partial R_{At} \partial R_{Bs}} = \frac{\partial^2 w_g}{\partial R_{Bs} \partial R_{At}}$；其二，平移不变性给出的行列和恒等式 $\sum_A \frac{\partial^2 w_g}{\partial R_{At} \partial R_{Bs}} = 0$ (对指标 $B$ 同样成立)；其三，与一阶梯度的中心差分比较

  $$
  \frac{\partial^2 w_g}{\partial R_{At} \partial R_{Bs}} \simeq \frac{1}{2h} \left[ \frac{\partial w_g}{\partial R_{Bs}} \Big|_{R_{At} + h} - \frac{\partial w_g}{\partial R_{Bs}} \Big|_{R_{At} - h} \right]
  $$

  偏差主要由差分截断误差 $O(h^2)$ 主导。
