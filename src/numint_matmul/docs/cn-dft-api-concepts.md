# Rust DFT 格点积分 API 设计概念

- API 设计原型、理念与简易实现 (简单矩阵乘法)。
- 可以实现 vxc, fxc, kxc 的计算。当前实现不保存格点，对内存压力较小，且性能尚可。
- 矩阵乘法后端使用 Faer，没有 BLAS 依赖，编译方便。但该程序仍然依赖 libcint，需要编译者在 `LIBRARY_PATH` 环境变量中作一些声明。
- 其测试比较充分；该测试框架可以作为后续代码修改的正确性测试基准。

本仓库的代码尽管目前没有进一步抽象为 Trait 接口；但我个人希望未来的 Trait 能够与本文档中描述的 API 设计概念相匹配。

对于性能，简要性能测试下 (移步文件夹 benches)，在 C22H14 TPSS0/def2-TZVP (nao = 766) 分子，表现尚可 (单步 DFT 格点计算相对于 PyPI 发行的 PySCF 快 20% 左右)。但也需要留意，**当前实现是非常简化的 BLAS-3，并没有针对性能进行细致的优化**。已知的问题是，当基组数特别大时 (超过 1500)、或并行数较大时 (超过 16)，性能会偏低；并行数超过 32 时耗时反而会上升。**该库的目的不是性能展示，而是 API 设计概念说明以及原型实现**。


# DFT 格点积分 API 设计概念

该文档将讨论 DFT 格点积分 API 设计中的一些核心概念。希望这些概念将有助于未来 DFT 格点积分的可扩展性、性能优化、API 调用友好性等方面的设计。

该文档也希望以相对比较一致的文本，厘清 DFT 程序所需要的名词、概念、公式。本文的说法未必是最正统的，但这确实是我在设计 API 时一贯的思路。

## 1. 通论：自洽场关键 API 设计

### 1.1 讨论前提：能量可分及其与密度矩阵的关系

计算任务的根本变量是基函数；这些基函数可能是原子轨道、平面波、实空间格点，不过原子轨道最常见。在该项目中，习惯称基函数为 AO (atomic orbital)，但这并不意味着我们只能处理原子轨道。

**能量是可以分解的**。譬如，对于 wB97X-V 泛函，其能量可以拆分为下述部分：

$$
E[\mathbf{D}] = E_\text{nuc-repl} + E_\text{kin} + E_\text{nuc} + E_\text{J} + E_\text{K} + E_\text{srK} + E_\text{xc} + E_\text{VV10}
$$

其中有一些要点是，在假定原子核不作为变量的前提下，**能量及其分项是密度矩阵 $\mathbf{D}$ 的函数**。这句话本身很简单，但隐含了下述推论或不太平常的反例：

- 密度矩阵定义为基函数系数矩阵 $C_{\mu i}$ 的函数：

    $$
    D_{\mu\nu} = \sum_i C_{\mu i} C_{\nu i}
    $$

    因此，能量及其分项也可以是基函数系数矩阵 $\mathbf{C}$ 的函数。

- 能量不能仅是系数矩阵 $\mathbf{C}$ 的函数。它必须要能写为密度矩阵 $\mathbf{D}$ 的显函数。这里的反例是各种 orbital-optimized post-SCF 方法。

- 能量与密度的关系必须严格满足 Hartree-Fock-Roothaan 方程：

    $$
    \mathbf{V}[\mathbf{D}] \mathbf{C} = \mathbf{S} \mathbf{C} \mathbf{\epsilon}
    $$
    
    其中，Fock 矩阵 $\mathbf{V}$ 是能量对密度矩阵 $\mathbf{D}$ 的梯度：

    $$
    \mathbf{V}[\mathbf{D}] = \frac{\partial E[\mathbf{D}]}{\partial \mathbf{D}}
    $$

    这里的反例是 constraint DFT (cDFT) 或 density-corrected DFT (dcDFT) 等方法。cDFT 包含约束项；dcDFT 的非自洽特性使得其 Fock 矩阵不再是能量的梯度。尽管我们这里写程序时仍然可以利用当前的 API，但在程序设计理念上，为了简化讨论，我们暂时不考虑这些方法。

- 原子结构不在考虑范围；我们只考虑电子结构 (以基函数表示的电子密度)。因此，对于原子核排斥能 $E_\text{nuc-repl}$、以及类似于 DFT-D3 的方法，我们认为它的能量项是常数。

同时，请留意，**本文的 Fock 矩阵以 $\mathbf{V} [\mathbf{D}]$ 表示，而非通常的 $\mathbf{F} [\mathbf{D}]$**。在下一小节将明确本文档记号。

### 1.2 核心问题：能量对密度矩阵导数

涉及到自洽场方法的计算问题，很多核心的技术问题，都涉及到如何计算能量对密度矩阵的导数。

我们这里需要作下述定义：

| 导数阶 | 变量 | 变量全写 | 惯用变量记号 |
|--|--|--|--|
| 1 | $\mathbf{V}$ | $\mathbf{V} [\mathbf{D}]$ | `fock`, `v`, `veff` |
| 2 | $\mathbf{F}$ | $\mathbf{F} [\mathbf{D}, \mathbf{R}]$ | `resp`, `f` |
| 3 | $\mathbf{K}$ | $\mathbf{K} [\mathbf{D}, \mathbf{R}^1, \mathbf{R}^2]$ | `k` |

- 本文档的程序惯用变量记号是 `v`, `f`, `k`，分别对应上述的 $\mathbf{V}$, $\mathbf{F}$, $\mathbf{K}$。

- 一阶导数定义

    $$
    V_{\mu \nu} = \frac{\partial E}{\partial D_{\mu \nu}}
    $$

    一阶导数在其他文献中，通常记为 $F_{\mu \nu}$。我们这里为了区分一阶、二阶导数，特意使用了不同的记号。

- 二阶导数定义

    $$
    F_{\mu \nu} = \sum_{\kappa \lambda} \frac{\partial^2 E}{\partial D_{\mu \nu} \partial R_{\kappa \lambda}} R_{\kappa \lambda}
    $$

    其中 $R_{\kappa \lambda}$ 是扰动密度矩阵；它在 TDDFT 中 (Casida 方程) 经常是激发态密度矩阵，在梯度问题中 (CP-HF/KS 方程) 是导数密度矩阵。

    在其他文献中，二阶导数通常记为 $\sum_{\kappa \lambda} A_{\mu \nu, \kappa \lambda} R_{\kappa \lambda}$，或 $G_{\mu \nu} [\mathbf{R}]$。

- 三阶导数定义

    $$
    K_{\mu \nu} = \sum_{\kappa \lambda} \sum_{\xi \zeta} \frac{\partial^3 E}{\partial D_{\mu \nu} \partial R^1_{\kappa \lambda} \partial R^2_{\xi \zeta}} R^1_{\kappa \lambda} R^2_{\xi \zeta}
    $$

    其中 $R^1_{\kappa \lambda}$ 和 $R^2_{\xi \zeta}$ 是两个扰动密度矩阵。

上面的所有定义，对所有可用于自洽场计算的能量分项都适用；且可以线性加和。譬如说，对于 wB97X-V 泛函，其 Fock 矩阵可以依葫芦画瓢写为：

$$
\mathbf{V} [\mathbf{D}] = \mathbf{V}_\text{nuc-repl} + \mathbf{V}_\text{kin} + \mathbf{V}_\text{nuc} + \mathbf{V}_\text{J} + \mathbf{V}_\text{K} + \mathbf{V}_\text{srK} + \mathbf{V}_\text{xc} + \mathbf{V}_\text{VV10}
$$

因此，对于任何一个能量分项而言，它都可以定义下述的程序接口：

```python
class EnergyComponent:
    def energy(self, dm0: np.ndarray) -> float:
    def get_fock(self, dm0: np.ndarray) -> np.ndarray:
    def get_2nd_resp(self, dm0: np.ndarray, dm1: np.ndarray) -> np.ndarray:
    def get_3rd_resp(self, dm0: np.ndarray, dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
```

### 1.3 技术考虑：性能与可扩展性

上述的接口设计，相信是可以实现所有重要的计算化学需求。但实际使用中，我们还需要作如下的考量与适配：

- **闭壳层与开壳层的密度矩阵维度有所差异**。闭壳层维度是 `[nao, nao]`，而开壳层维度是 `[nao, nao, 2]` (col-major)。
- 密度矩阵，特别是扰动密度矩阵，**经常是多个**而非单个。因此，对于闭壳层，传入的 `dm1` 与 `dm2` 一般要允许是 3-dim 张量、或列表的 2-dim 矩阵。传入的 `dm0` 大多数情况下是单个矩阵。
- 密度矩阵通常是低秩的。它是由占据轨道系数构成的，而占据数 $n_\text{occ}$ 通常远小于基组数 $n_\text{AO}$。即使是 CP-KS 方程求解中所需要的扰动密度 (不同于自洽场密度)，它从构造上 (或者用 SVD 等方式分解上) 也可以分解为两个 $n_\text{AO} \times n_\text{occ}$ 的矩阵的乘积。因此**应该尽量利用密度矩阵的低秩结构**，以降低计算量。
    - 另一方面，对于特别大的分子，密度矩阵本身可能有足够的稀疏程度。密度矩阵零值的稀疏性与低秩结构不能 (至少难以) 同时利用；某些情况下，密度矩阵本身的稀疏性也可以利用。但我们关注的原子体系如果不超过 100 个原子，那么密度矩阵的低秩结构通常更重要。
    - 从 API 设计的角度而言，低秩性质的利用，等同于在计算函数中传入占据轨道系数矩阵：
        
        ```python
        def get_fock_by_occ(self, occ_coeff: np.ndarray) -> np.ndarray:
        ```

        具体的接口形式可能会有一些变化，但核心思想是，**API 设计应该允许直接传入占据轨道系数矩阵**，以利用密度矩阵的低秩结构。我们会在后面看到，在 DFT 格点积分中，我们是如何具体地利用这一点。


## 2. DFT 格点积分 API 设计

### 2.1 与其他能量分项的比较

DFT 格点积分，与其他计算化学的能量分项贡献，有相同之处，也有不同之处。

相同点有

- 上述总论的概念，都适用于 DFT 格点积分。
- DFT 同样涉及到是否能利用占据轨道进行优化的问题。由于引入占据轨道与否，不只是会对程序实现细节产生影响，也会将 API 设计的问题变得复杂化。
- DFT 格点积分的计算复杂度相对较低，但计算量经常不算太小，特别是对于小到中等体系而言。因此，在设计 API 时，仍然需要考虑性能因素。

不同点有

- 对于 J/K 计算，一来能量是密度矩阵的二次函数 (因而 J/K 能量不存在三阶项 $\mathbf{K}$)，二来 J/K 的二阶项计算方式与一阶项非常相似。因此，尽管 J/K 计算量大、对其作近似仍然是现在电子结构程序的研究重点之一，但它的 API 设计相对简单。
- DFT 的困难在于，其能量是是密度的函数 (无法 Taylor 截断到有限阶)。且越高阶，计算量越大。如果不作合理的程序设计，其程序实现难度也会很大。

### 2.2 DFT 格点计算步骤：密度生成、泛函计算、能量导数矩阵组装

DFT 格点积分的计算过程可以清晰地分为三个步骤。这三个步骤的输入输出之间有明确的依赖关系，但它们在计算特征上有显著差异。我们在这里对闭壳层（RKS）情形逐一说明，并区分 $\text{RHO}$ (LDA)、$\text{SIGMA}$ (GGA)、$\text{TAU}$ (mGGA) 三种密度类型（我们不考虑 $\text{LAPL}$ 型 mGGA）。

#### 步骤一：密度格点生成 (eval_rho)

以密度矩阵 $D_{\mu\nu}$ 为输入，原子轨道格点 $\varphi_{\mu g}$ 及其空间导数 $\varphi_{\mu g}^t$ (其中 $t \in \{x, y, z\}$) 为中间量，生成密度格点。定义密度变量

$$
\xi_g[\mathbf{D}] := (\rho_g, \rho_g^x, \rho_g^y, \rho_g^z, \tau_g)
$$

对于闭壳层，该变量记为 $\rho_g^\xi$ 在程序中是 $[n_\text{grid}, n_\text{var}]$ 的二维数组。对于开壳层，该变量记为 $\rho_g^{\xi \sigma}$ 即增加一个自旋维度，在程序中是 $[n_\text{grid}, n_\text{var}, 2]$ 的三维数组。

各分量计算公式如下：

- **密度** $\rho_g$ (所有类型都需要)

    $$\rho_g = \sum_{\mu\nu} \varphi_{\mu g} D_{\mu\nu} \varphi_{\nu g}$$

    算法实现为一步矩阵乘法加一步数乘约化：

    $$\tilde{\varphi}_{\mu g} = \sum_\nu D_{\mu\nu} \varphi_{\nu g} \quad \text{(GEMM)}$$

    $$\rho_g = \sum_\mu \varphi_{\mu g} \tilde{\varphi}_{\mu g} \quad \text{(memory bounded)}$$

- **密度梯度** $\rho_g^t$ (GGA / mGGA 需要)

    $$\rho_g^t = 2 \sum_{\mu\nu} \varphi_{\mu g}^t D_{\mu\nu} \varphi_{\nu g}$$

    由于 $\tilde{\varphi}_{\mu g}$ 已经在 $\rho_g$ 的计算中得到，因此每个梯度分量只需额外一步数乘约化 $\sum_\mu \varphi_{\mu g}^t \tilde{\varphi}_{\mu g}$，乘以系数 2。三个分量总共的额外计算量为 memory bounded。

- **动能密度** $\tau_g$ (mGGA 需要)

    $$\tau_g = \sum_{t,\mu\nu} \frac{1}{2} \varphi_{\mu g}^t D_{\mu\nu} \varphi_{\nu g}^t$$

    这需要对每个 $t$ 计算一步新的矩阵乘法 $\tilde{\varphi}_{\mu g}^{(t)} = \sum_\nu D_{\mu\nu} \varphi_{\nu g}^t$（乘以系数 1/2），再作数乘约化 $\sum_\mu \varphi_{\mu g}^t \tilde{\varphi}_{\mu g}^{(t)}$。三步 GEMM 的 FLOPs 约 $6 n_\text{basis}^2 n_\text{grid}$。

**低秩优化**：当密度矩阵具有低秩结构 $D_{\mu\nu} = \sum_i C_{\mu i} C_{\nu i}$ 时（$n_\text{occ} \ll n_\text{basis}$），可以通过占据轨道系数 $C_{\mu i}$ (bra-ket 形式) 将计算量降低：

$$\varphi_{i g} = \sum_\mu \varphi_{\mu g} C_{\mu i} \quad \text{(GEMM)}$$

$$\rho_g = \sum_i \varphi_{i g}^{(\text{bra})} \varphi_{i g}^{(\text{ket})} \quad \text{(memory bounded)}$$

对于 def2-TZVP 级别的 3-$\zeta$ 基组，$n_\text{basis} / n_\text{occ} \sim 10$，因此计算量可降低约一个数量级。

各密度类型下，密度格点生成的计算量总结如下：

| 密度类型 | 变量数 $n_\text{var}$ | AO 导数阶 | AO 分量数 $n_\text{comp}$ | GEMM 数 (DM输入) |
|--|--|--|--|--|
| RHO (LDA) | 1 | 0 | 1 | 1 |
| SIGMA (GGA) | 4 | 1 | 4 | 1 |
| TAU (mGGA) | 5 | 1 | 4 | 4 |

密度格点生成是计算量仅次于响应矩阵生成的步骤。

#### 步骤二：泛函计算 (eval_xc)

将密度格点 $\xi_g[\mathbf{D}]$ 代入到密度泛函 $f(\rho, \nabla\rho, \tau)$ 及其各阶偏导数，得到泛函输出量。这部分的计算量是 $O(n_\text{grid})$，不是计算瓶颈。

泛函输出量的维度定义如下：

- 一阶导数（有效势，用于 vxc）：$f_{g}^{\xi}$，维度 $[n_\text{grid}, n_\text{var}]$
- 二阶导数（有效核，用于 fxc）：$f_{g}^{\xi \xi'}$，维度 $[n_\text{grid}, n_\text{var}, n_\text{var}]$
- 三阶导数（有效核，用于 kxc）：$f_{g}^{\xi \xi' \xi''}$，维度 $[n_\text{grid}, n_\text{var}, n_\text{var}, n_\text{var}]$

**重要设计选择**：本程序使用密度梯度分量 $\rho_g^t$ 作为泛函的基本变量，而非 $\gamma = |\nabla\rho|^2$。这是因为 $\gamma$ 是密度矩阵的二阶量（对 $\gamma$ 作密度矩阵的额外导数不为零），而 $\nabla\rho$ 是密度矩阵严格的一阶量。这使得后续公式推导与程序实现更为简单。代价是格点维度增大（自旋非极化 LDA/GGA/mGGA 从 1/2/3 增加到 1/4/5，并且我们的程序实现中没有利用 ），但由于 DFT 格点积分的瓶颈是 GEMM 运算而非格点维度，这一代价在通常不反应在真正的计算瓶颈里。

从 $\gamma$ 到 $\rho_t$ 的变换遵循链式法则：

$$\frac{\partial(f\rho)}{\partial\rho_t} = \frac{\partial(f\rho)}{\partial\gamma} \frac{\partial\gamma}{\partial\rho_t} = 2 f^\gamma \rho_t, \quad t \in \{x, y, z\}$$

#### 步骤三：能量导数矩阵组装 (contract_ao_wv)

将格点数乘量 $w_g f_\text{eff}$ 与原子轨道格点 $\varphi_{\mu g}$ 作缩并，得到能量导数矩阵（Fock 矩阵的 XC 贡献）。

**一阶响应 (vxc)**。对于闭壳层情形：

$$V_{\mu\nu}^{\text{xc}} = \sum_g w_g f_g^\rho \varphi_{\mu g} \varphi_{\nu g} \quad \text{(LDA)}$$

$$V_{\mu\nu}^{\text{xc}} \leftarrow \sum_{t,g} w_g f_g^{\rho_t} \varphi_{\mu g}^t \varphi_{\nu g} + \text{swap}(\mu, \nu) \quad \text{(GGA)}$$

$$V_{\mu\nu}^{\text{xc}} \leftarrow \sum_{t,g} \frac{1}{2} w_g f_g^\tau \varphi_{\mu g}^t \varphi_{\nu g}^t \quad \text{(mGGA)}$$

上述表达式的共同结构是：**左矢 $\varphi_{\mu g}^{(\text{lhs})}$、右矢 $\varphi_{\nu g}^{(\text{rhs})}$、格点数乘量 $w_g f_\text{eff}$，最终对格点指标 $g$ 求和**。不同密度类型下，左矢、右矢和数乘量的具体内容不同，但结构一致。

**二阶响应 (fxc)**。引入广义密度变量 $\xi$ 后，fxc 的表达式为

$$F_{\mu\nu}[\mathbf{R}] = \sum_g \sum_\xi w_g \left(\sum_{\xi'} f_g^{\xi\xi'} \xi'_g[\mathbf{R}]\right) \frac{\partial \xi_g[\mathbf{D}]}{\partial D_{\mu\nu}}$$

其中 $\sum_{\xi'} f_g^{\xi\xi'} \xi'_g[\mathbf{R}]$ 是格点空间上的缩并，得到关于角标 $\xi$ 的格点量；随后的 $\sum_\xi (\cdots) \frac{\partial \xi_g}{\partial D_{\mu\nu}}$ 的处理与 `eval_vxc` 完全一致。这意味着 fxc 的格点缩并步骤可以复用 vxc 的 `contract_ao_wv` 函数。

**三阶响应 (kxc)** 类似地，在格点空间缩并 $\sum_{\xi'\xi''} f_g^{\xi\xi'\xi''} \xi'_g[\mathbf{R}'] \xi''_g[\mathbf{R}'']$ 后，同样复用 `contract_ao_wv`。

**对称化策略与系数**。本程序在 `contract_ao_wv` 中先计算非对称的半矩阵，再通过 $V_{\mu\nu} \leftarrow V_{\mu\nu} + V_{\nu\mu}$ 实现对称化。这导致了不同密度类型下系数的差异：

- LDA 贡献：先乘以 0.5 计算半矩阵 $\frac{1}{2} \varphi_{\mu g}^T (w_g f^\rho \varphi_{\nu g})$，对称化后恰好恢复为完整值。
- GGA 贡献：$\varphi_{\mu g}^t (w_g f^{\rho_t} \varphi_{\nu g})$ 的系数为 1.0（非对称项），对称化后自然得到 $\varphi_{\mu g}^t (w_g f^{\rho_t} \varphi_{\nu g}) + \varphi_{\nu g}^t (w_g f^{\rho_t} \varphi_{\mu g})$。
- mGGA 贡献：$\varphi_{\mu g}^t (w_g f^\tau \varphi_{\nu g}^t)$ 的系数为 0.25，对称化后恰好恢复为 $\frac{1}{2}\sum_t$（因为 $\mu, \nu$ 对称性与 $t$ 指标无关）。

**计算瓶颈**。响应矩阵生成是 DFT 格点积分中计算量最大的步骤。对于 vxc，FLOPs 量级为 $O(n_\text{basis}^2 n_\text{grid} n_\text{var})$；对于 fxc，由于有 $n_\text{set}$ 个扰动密度矩阵，量级为 $O(n_\text{basis}^2 n_\text{grid} n_\text{var}^2 n_\text{set})$。密度格点生成次之。泛函计算可以忽略。


### 2.3 总体设计：分离泛函计算与格点缩并

从上一节的分析可以看出，三个计算步骤在计算特征上有显著差异：

- **密度格点生成**与**能量导数矩阵组装**都是矩阵运算（GEMM + 数乘约化），与泛函的具体形式无关；
- **泛函计算**是逐格点的运算（$O(n_\text{grid})$），与原子轨道的基组结构无关。

因此，我们将泛函计算与格点缩并（在本程序中称为 NIMatmul）分离为独立的模块。这一分离带来了以下好处：

1. **格点缩并函数可以接受"有效势" (eff\_pot) 作为输入**，而非泛函的原始输出。有效势是泛函输出 $f_g^{\xi}$ 与密度格点 $\xi'_g[\mathbf{R}]$（在二阶以上时）作格点空间缩并后的格点权重向量。这使得 `contract_ao_wv` 系列函数完全不依赖 libxc，可以独立测试与优化。

2. **泛函计算模块可以独立替换**。当前使用 libxc，但未来可以接入 xcfun、机器学习泛函或其他泛函引擎，只要其输出格式符合有效势的约定即可。

3. **数据流清晰**。完整的数据流如下：

    ```
    // vxc
    dm → [eval_rho] → rho → [eval_xc] → vxc_eff → [contract_ao_wv] → vxc
    
    // fxc
    dm + dm1 → [eval_rho] → rho, rho1 → [eval_xc] → fxc_eff → [contract with rho1] → fxc_eff_contracted → [contract_ao_wv] → fxc
    
    // kxc
    dm + dm1 + dm2 → [eval_rho] → rho, rho1, rho2 → [eval_xc] → kxc_eff → [contract with rho1, rho2] → kxc_eff_contracted → [contract_ao_wv] → kxc
    ```

    对于一阶 (vxc)，泛函输出直接就是有效势。对于二阶 (fxc) 和三阶 (kxc)，需要在格点空间先作缩并 $\sum_{\xi'} f^{\xi\xi'} \xi'_g[\mathbf{R}]$（或 $\sum_{\xi'\xi''} f^{\xi\xi'\xi''} \xi'_g[\mathbf{R}'] \xi''_g[\mathbf{R}'']$），得到有效势后再进入 `contract_ao_wv`。

4. **低秩优化可以在 eval_rho 层面实现**，不影响 eval_xc 和 contract_ao_wv 的接口。本程序提供了四种密度格点生成方式（见 2.5 节），以支持不同场景下的低秩优化需求。


### 2.4 三层架构：纯函数算法、DFT 公共接口、Fock 接口

本程序采用三层架构设计。从底层到顶层分别是：

#### 纯函数算法层（最底层）

这一层包含 `pure_eval_rho.rs` 和 `pure_xcpot.rs` 等文件中的函数。它们的特点是：

- **无状态**：函数不持有任何 `self` 或隐藏状态，所有输入通过参数显式传入。
- **参数简单**：输入为 tensor views 和枚举参数（如 `NIDenType`），输出写入预分配的 buffer。最好不要有过于复杂的类型 (譬如完整的 grids 结构)。理想情况下，这类函数也是容易 export 到 C API 的。
- **数据结构扁平**：不涉及格点分批、AO 缓存等复杂逻辑。

纯函数也是**性能热点**所在。当前项目的大多数纯函数都有 `_naive`（串行参考）和优化两个版本。由于参数表完全显式，这一层可以被独立测试、替换或进一步优化，而不影响上层接口。

#### DFT 公共接口层（中间层）

这一层包含 `NIMatmul` struct（`nimatmul.rs`）和泛函计算桥接（`libxc_wrap.rs`、`xc_deriv.rs`）。它们的特点是：

- **有状态**：`NIMatmul` 管理 AO 缓存、格点分批参数（`nchunk`, `nbatch`）、积分引擎等。
- **参数较复杂**：方法调用涉及缓存策略、分批逻辑、泛函引擎选择等。

这一层是**可能发生变化的层**。例如，更换积分引擎、改变格点存储格式（从稠密 $\varphi_{\mu g}$ 到 Psi4 blocking 或稀疏格式）、调整缓存策略等，都主要在这一层实现。但重要的是，**这一层的变动不应影响纯函数层和 Fock 接口层**。

目前，非梯度计算（vxc）的公共接口已比较完备；fxc 和 kxc 的公共接口也已实现，但 bra-ket 形式的 fxc（bra\_trans）目前仅支持特定场景。

#### Fock 接口层（最上层）

这一层包含 `xcpot_fock_naive.rs` 中的函数。它们的特点是：

- **完整计算流程**：将三层计算步骤串联为完整的能量/Fock 计算。
- **格点分批循环**：对格点按 `nbatch` 分批，每批内部执行 eval\_rho → eval\_xc → contract\_ao\_wv 三步，累积结果。

这一层面向最终用户，其接口形式与 PySCF 的 `dft.numint.nr_rks` / `nr_uks` 类似。只要纯函数层和公共接口层的 API 不变，Fock 接口层可以保持稳定。

**三层架构的核心意义**在于：性能优化应集中在纯函数层（`contract_ao_wv` 等热点函数），数据结构的调整、或非平凡的性能优化问题发生在中间层，而顶层用户接口保持不变。这使得不同层面的开发工作可以解耦进行。

在未来，我们也会将 DFT 层与 Fock 层尝试抽象出 Trait，以进一步允许各类数据结构实现相同的接口，从而增强灵活性和可扩展性。


### 2.5 DFT 公共接口

DFT 公共接口是三层架构中的中间层，负责管理状态、缓存、格点分批，并将上层调用分发到纯函数算法。

这里的公共接口，未来可以考虑抽象到 Trait，以允许不同的数据结构实现相同的接口。但目前我们先以 `NIMatmul` struct 的方法为主线，说明公共接口的设计。

#### 一些重要公式记号约定

| 符号 | 大小 | 惯用变量名 | 说明 |
|--|--|--|--|
| $\mathbb{A}$ | $n_\text{set}$ | `nset` | 密度矩阵列表长度 (通常对应到性质计算数量，有时也代表自旋) |
| $\xi$ | $n_\text{var}$ | `nvar` | 密度变量数 (RHO: 1, SIGMA: 4, TAU: 5) |
| | $n_\text{comp}$ | `ncomp` | 原子轨道格点分量数 (RHO: 1, SIGMA: 4, TAU: 4) |
| $\mu, \nu$ | $n_\text{AO}$ | `nao` | 原子轨道数 |
| $g$ | $n_\text{grid}$ | `ngrids` | 格点数 |
| $i$ | $n_\text{occ}$ | `nocc` | 占据轨道数 |
| $\sigma$ | 2 | `nspin` | 自旋数 |

#### NIMatmul struct

`NIMatmul` 是格点驱动的主结构体，定义在 `nimatmul.rs` 中。其关键字段为：

- `cint: CInt` — 积分引擎（libcint 包装）
- `coords: Vec<[f64; 3]>` — 格点坐标
- `weights: Vec<f64>` — 格点权重
- `cache_tensor: HashMap<String, TsrCow>` — AO 缓存，以导数阶为键（如 `"ao_deriv0"`, `"ao_deriv1"`）
- `nchunk: usize` — 并行分块大小（默认 384，通常对应 micro-kernel KC 维度）
- `nbatch: usize` — 内存分批大小（默认 $384 \times 4 \times n_\text{threads}$）

**格点分批 vs 格点分块**：`nbatch` 控制内存用量（完整 AO 张量 `[ngrids, nao, ncomp]` 可能过大），`nchunk` 控制并行粒度。关系为 full-grid > batch > chunk > per-grid = 1。

**AO 缓存策略**：`get_cached_ao(deriv)` 方法在需要时计算 AO 并缓存。如果高阶导数已缓存，低阶导数可以从中切片取出，避免重复计算 (这也是为何我们需要用 `TsrCow` 即 copy-on-write 类型张量的原因)。`prepare_ao(deriv)` 通过 libcint 的 `eval_gto` 计算 AO 格点，输出形状为 `[ngrids, nao, ncomp]`。

#### NIMatmul 密度生成

| 方法| 说明 | 常见情景 |
|--|--|--|
| `make_rho_from_dm` | 从密度矩阵 $D_{\mu \nu}^\mathbb{A}$ 生成密度 | 基础功能，post-SCF 的驰豫密度计算 |
| `make_rho_from_homogeneous_braket` | 从同构左右系数 $C_{\mu i}^\mathbb{A}$ 生成密度 | 自洽场 |
| `make_rho_from_one_bra_mult_ket` | 同左系数 $C_{\mu i}^\text{bra}$，多右系数 $C_{\mu i}^{\mathbb{A}, \text{ket}}$ | 梯度性质 |
| `make_rho_from_mult_bra_mult_ket` | 多作系数 $C_{\mu i}^{\mathbb{A}, \text{bra}}$，多右系数 $C_{\mu i}^{\mathbb{A}, \text{ket}}$ |  |
  
上述函数的输出均为

- 密度格点 $\rho_g^{\xi \mathbb{A}}$，维度 $[n_\text{grid}, n_\text{var}, n_\text{set}]$。

具体计算的表达式随 RHO/SIGMA/TAU 而异，详见前文 2.2 节。

上述函数的输入需要作说明如下：

- **make_rho_from_dm**
  - `dm_list`：矩阵列表 $D_{\mu \nu}^\mathbb{A}$，列表长度 $n_\text{set}$，维度 $[n_\text{AO}, n_\text{AO}]$。目前的程序实现中，**必须要求是对称的**。
  - 对于闭壳层自洽场计算，必须要将单个密度转换为长度为 1 的密度列表。对于开壳层计算，可以将性质变量 $\mathbb{A}$ 视作自旋变量 $\sigma$ 以进行计算。 
- **make_rho_from_homogeneous_braket**
  - `bra`：同构左右系数列表 $C_{\mu i}^\mathbb{A}$，列表长度 $n_\text{set}$，维度 $[n_\text{AO}, n_\text{occ}]$。
  - 不要求列表中的每个占据数是相同的。即可以传入开壳层 $\alpha, \beta$ 电子数不同的左系数矩阵。
  - 对于闭壳层情景，占据数通常是 2。程序要么需要用户在传入左系数 $C_{\mu i}$ 时乘以 $\sqrt{2}$，要么用户后续手动将输出乘以 2。我们倾向于希望用户采用前者的策略。
- **make_rho_from_one_bra_mult_ket**
  - `bra`：单个左系数列表 $C_{\mu i}^\text{bra}$，维度 $[n_\text{AO}, n_\text{occ}]$。
  - `ket_list`：多个右系数 $C_{\mu i}^{\mathbb{A}, \text{ket}}$，列表长度 $n_\text{set}$，维度 $[n_\text{AO}, n_\text{occ}]$。占据数 $n_\text{occ}$ 必须与 `bra` 中的占据数相同。
  - 这一函数的设计初衷是为了性质计算中，在求解 TD/CP-KS 方程时，涉及到的 $X_{ai}^\mathbb{A}$ 在代入到 DFT 计算中时需要先作原子轨道转换 $D_{\mu \nu}^\mathbb{A} = \sum_{ai} C_{\mu i}^{\text{bra}} X_{ai}^\mathbb{A} C_{\nu a}^\text{ket}$。考虑到占据轨道数通常远小于原子轨道数，我们确实可以传入 $C_{\mu i}^\text{bra}$ 以节省计算量；但另一边比较合适的做法是作轨道半转换 $\tilde{C}_{\nu i}^{\mathbb{A}, \text{ket}} = \sum_a X_{ai}^\mathbb{A} C_{\nu a}^\text{ket}$，并将该半转换后的系数作为右系数传入函数参数。
  - 对于开壳层情况，该函数不能同时处理 $\alpha$ 与 $\beta$ 自旋。需要分两次计算。
- **make_rho_from_mult_bra_mult_ket**
  - `bra_list`：多个左系数列表 $C_{\mu i}^{\mathbb{A}, \text{bra}}$，列表长度 $n_\text{set}$，维度 $[n_\text{AO}, n_\text{occ}]$。
  - `ket_list`：多个右系数列表 $C_{\mu i}^{\mathbb{A}, \text{ket}}$，列表长度 $n_\text{set}$，维度 $[n_\text{AO}, n_\text{occ}]$。
  - 两个列表长度需要相同，占据数需要一一对应。

#### 泛函计算桥接 (xceff)

`libxc_wrap.rs` 提供 `libxc_eval_eff` 函数，将 LibXC 的原始输出转换为有效势格式。

输入的密度维度是 `[ngrids, nvar]` (闭壳层) 或 `[ngrids, nvar, 2]` (开壳层)。用户需要在生成 `LibXCFunctional` 示例时同时指定其自旋，我们的程序是以该实例来判断是否是开闭壳层。

输出的有效势维度则根据导数阶数不同而不同。

有效势的输出维度（闭壳层）为：
- deriv=0: `[ngrids]` (exc, $f_g$)
- deriv=1: `[ngrids, nvar]` (vxc_eff, $f_g^\xi$)
- deriv=2: `[ngrids, nvar, nvar]` (fxc_eff, $f_g^{\xi \xi'}$)
- deriv=3: `[ngrids, nvar, nvar, nvar]` (kxc_eff, $f_g^{\xi \xi' \xi''}$)

开壳层则在 `nvar` 前插入自旋维度：
- deriv=0: `[ngrids]` (exc, $f_g$)
- deriv=1: `[ngrids, nvar, 2]` (vxc_eff, $f_g^{\xi_\sigma}$)
- deriv=2: `[ngrids, nvar, 2, nvar, 2]` (fxc_eff, $f_g^{\xi_\sigma \xi'_{\sigma'}}$)
- deriv=3: `[ngrids, nvar, 2, nvar, 2, nvar, 2]` (kxc_eff, $f_g^{\xi_\sigma \xi'_{\sigma'} \xi''_{\sigma''}}$)

#### NIMatmul XC 势矩阵组装

| 方法 | 说明 |
|--|--|
| `make_vxc_pot_with_eff` | 一阶 XC 势 |
| `make_fxc_pot_with_eff` | 二阶 XC 核 |
| `make_kxc_pot_with_eff` | 三阶 XC 核 |
| `make_rks_fxc_pot_with_eff_bra_trans` | 二阶 XC 核（bra-transformed，低秩优化） |
| `make_uks_fxc_pot_with_eff_bra_trans` | 二阶 XC 核（bra-transformed，低秩优化） |

上述函数的输入输出需要作说明如下：

- **make_vxc_pot_with_eff**
  - 输入：`vxc_eff`，维度 `[ngrids, nvar]` (闭壳层) 或 `[ngrids, nvar, 2]` (开壳层)。这是泛函计算模块的输出，泛函所使用密度一般是自洽场密度。
  - 输出：vxc 势矩阵，维度 `[nao, nao]` (闭壳层) 或 `[nao, nao, 2]` (开壳层)。
  - 计算过程：
    
    $$
    V_{\mu \nu}^\text{xc} = \sum_g \sum_\xi w_g f_g^\xi \frac{\partial \xi_g}{\partial D_{\mu\nu}}
    $$

    其中
    - $\partial \xi_g / \partial D_{\mu\nu}$ 与其他组分的乘积是通过 `contract_ao_wv` 函数实现的；由于 $\xi_g$ 是密度矩阵的一阶函数，因此 $\partial \xi_g / \partial D_{\mu\nu}$ 是密度矩阵无关量，只与轨道格点 $\phi_{g \mu}$ 及其梯度有关；这是 `NIMatmul` (作为 `&mut self` 变量) 会考虑的事情。
    - $w_g$ 同样是 `NIMatmul` (作为 `&mut self` 变量) 会考虑的事情。因此不作为参数传入。

- **make_fxc_pot_with_eff**
  - 输入：`fxc_eff`，维度 `[ngrids, nvar, nvar]` (闭壳层) 或 `[ngrids, nvar, 2, nvar, 2]` (开壳层)。
  - 输入：`rho1`，维度 `[ngrids, nvar, nset]` (闭壳层) 或 `[ngrids, nvar, 2, nset]` (开壳层)。
  - 输出：fxc 势矩阵，维度 `[nao, nao, nset]` (闭壳层) 或 `[nao, nao, 2, nset]` (开壳层)。
  - 计算过程：
    
    $$
    F_{\mu\nu}^\text{xc} [\mathbf{R}^\mathbb{A}] = \sum_g \sum_\xi \left( w_g \sum_{\xi'} f_g^{\xi\xi'} \xi'_{g}[\mathbf{R}^\mathbb{A}] \right) \frac{\partial \xi_g}{\partial D_{\mu\nu}}
    $$

- **make_kxc_pot_with_eff**
  - 输入：`kxc_eff`，维度 `[ngrids, nvar, nvar, nvar]` (闭壳层) 或 `[ngrids, nvar, 2, nvar, 2, nvar, 2]` (开壳层)。
  - 输入：`rho1`，维度 `[ngrids, nvar, nset1]` (闭壳层) 或 `[ngrids, nvar, 2, nset1]` (开壳层)。
  - 输入：`rho2`，维度 `[ngrids, nvar, nset2]` (闭壳层) 或 `[ngrids, nvar, 2, nset2]` (开壳层)。
  - 输出：kxc 势矩阵，维度 `[nao, nao, nset1]` (闭壳层) 或 `[nao, nao, 2, nset1, nset2]` (开壳层)。
  - 计算过程：
    
    $$
    K_{\mu\nu}^\text{xc} [\mathbf{R}'^{\mathbb{A}}, \mathbf{R}''^{\mathbb{A}}] = \sum_g \sum_\xi \left( w_g \sum_{\xi'} k_g^{\xi\xi'\xi''} \xi'_{g}[\mathbf{R}^\mathbb{A}] \xi''_{g}[\mathbf{R}''^{\mathbb{A}}] \right) \frac{\partial \xi_g}{\partial D_{\mu\nu}}
    $$

- **make_rks_fxc_pot_with_eff_bra_trans**
  - 输入：`fxc_eff`，维度 `[ngrids, nvar, nvar]` (闭壳层)。
  - 输入：`rho1`，维度 `[ngrids, nvar, nset]` (闭壳层)，其中 `nset` 是 bra-ket 形式中 ket 的数量。
  - 输入：`bra`，维度 `[nao, nocc]` (闭壳层)。
  - 输出：fxc 势矩阵，维度 `[nao, nocc, nset]` (闭壳层)。
  - **该函数仅用于闭壳层**。这是因为开壳层的的输入 `bra` 与输出不能是单个更高一阶的、带有自旋维度的张量，而必须是两个单独的 $\alpha$ 与 $\beta$ 张量。由于类型不同，因此在 Rust 等强类型语言中无法通过一个函数实现。
  - 函数意义：该函数实际上就是计算了

    $$
    F_{\mu i}^\text{xc} [\mathbf{R}^\mathbb{A}] = \sum_\nu F_{\mu\nu}^\text{xc} [\mathbf{R}^\mathbb{A}] C_{\nu i}
    $$

    但实际上，一般来说性能最优的实现模式是对 $\partial \xi_g / \partial D_{\mu\nu}$ 与 $C_{\nu i}$ 的乘积进行低秩优化。该优化函数在程序中是 `contract_ao_wv_bra`。

    之所以需要设计这样的函数，是因为 TD/CP-KS 方程中经常出现下述形式的计算问题：

    $$
    A_{a i, b j}^\text{xc} R_{b j} = \sum_{\mu \nu} C_{\mu a} C_{\nu i} F_{\mu\nu}^\text{xc} [\mathbf{R}]
    $$

    而先缩并占据轨道得到 $F_{\mu i}^\text{xc} [\mathbf{R}]$ 是性能更好的做法，因此上式化为
    
    $$
    A_{a i, b j}^\text{xc} R_{b j} = \sum_{\mu i} C_{\mu a} F_{\mu i}^\text{xc} [\mathbf{R}]
    $$

    得到 $F_{\mu i}^\text{xc} [\mathbf{R}]$ 这一步只是半转换。需要留意，全转换在 $n_\mathrm{set} = 1$ 的计算代价基本不会减少，在 $n_\mathrm{set} > 1$ 的情形时计算量会上升；因此是要处理 CP-KS 的情况 (有多个性质矩阵要计算)，全转换不如半转换。因此我们在公共接口层仅提供了半转换的函数。

- **make_uks_fxc_pot_with_eff_bra_trans**
    - 输入：`fxc_eff`，维度 `[ngrids, nvar, 2, nvar, 2]` (开壳层)。
    - 输入：`rho1`，维度 `[ngrids, nvar, 2, nset]` (开壳层)，其中 `nset` 是 bra-ket 形式中 ket 的数量。
    - 输入：`bra`，两个自旋的左系数，维度 `[nao, nocc_alpha]` 与 `[nao, nocc_beta]`。
    - 输出：两个自旋的 fxc 势矩阵，维度 `[nao, nocc_alpha, nset]` 与 `[nao, nocc_beta, nset]` (开壳层)。
    - 函数意义同上，但仅适用于开壳层。由于输入与输出类型不同，因而不能与闭壳层的函数合并。


### 2.6 Fock 接口

Fock 接口是三层架构的最上层，面向最终用户。当前实现在 `xcpot_fock_naive.rs` 中。

该接口的实现，目前需要同时引入格点积分程序 `NINumint` 与 DFT 泛函计算两套模块。

包含 8 个公共函数，构成 4 对（RKS/UKS × DM/bra-ket × vxc/fxc）：

**vxc 函数（一阶响应）**：

| 函数 | 输入 | 输出 | PySCF 对应 |
|--|--|--|--|
| `compute_rks_vxc_from_dm_naive` | DM `[nao, nao]` | `(nelec, exc, vxc[nao,nao])` | `dft.numint.nr_rks` |
| `compute_rks_vxc_from_homogenous_bra_naive` | bra `[nao, nocc]` | 同上 | 同上 (tagged mo_coeff) |
| `compute_uks_vxc_from_dm_naive` | DM `[nao, nao, 2]` | `(nelec, exc, vxc[nao,nao,2])` | `dft.numint.nr_uks` |
| `compute_uks_vxc_from_homogenous_bra_naive` | bra `[alpha, beta]` | 同上 | 同上 |

**fxc 函数（二阶响应）**：

| 函数 | 输入 | 输出 | PySCF 对应 |
|--|--|--|--|
| `compute_rks_fxc_from_dm_naive` | dm0 + dm1_list | `fxc[nao, nao, nset]` | `dft.numint.nr_rks_fxc` |
| `compute_rks_fxc_from_braket_naive` | bra0, bra1, ket1_list | 同上 | 同上 (tagged) |
| `compute_uks_fxc_from_dm_naive` | dm0 + dm1_list | `fxc[nao, nao, 2, nset]` | `dft.numint.nr_uks_fxc` |
| `compute_uks_fxc_from_braket_naive` | bra0, bra1, ket1_list | 同上 | 同上 |

**通用算法模式**（以 `compute_rks_vxc_from_dm_naive` 为例）：

```rust
for start in (0..ngrids).step_by(nbatch) {
    // 1. 创建分批 NIMatmul
    let ni_cur = NIMatmul::new(&cint, &coords[start..stop], &weights[start..stop]);
    // 2. 密度格点生成
    let rho = ni_cur.make_rho_from_dm(&[dm0], den_type);
    // 3. 泛函计算
    let [exc_eff, vxc_eff] = libxc_eval_eff(xc_func, rho, deriv=1, par=true);
    // 4. 累积能量与电子数
    nelec += (weights * rho[0]).sum();
    exc += (exc_eff * weights * rho[0]).sum();
    // 5. XC 势矩阵组装
    let vxc_batch = ni_cur.make_vxc_pot_with_eff(vxc_eff, den_type, spin);
    vxc += vxc_batch;
}
```

对于 fxc，需要额外计算 rho1（扰动密度的格点值），将 fxc_eff 与 rho1 在格点空间缩并后得到有效势，再调用 `make_fxc_pot_with_eff`。bra-ket 形式的 fxc 使用 `make_rho_from_homogeneous_braket` 计算 rho0，`make_rho_from_one_bra_mult_ket` 计算 rho1，并使用 `make_fxc_pot_with_eff_bra_trans` 进行低秩优化的矩阵组装。

函数名中的 `_naive` 表示这是参考实现，未来可能有其他替换，但接口形式应保持基本一致。


### 2.7 程序优化考量

本程序的核心目标是 API 设计概念说明，而非性能展示。但三层架构的分离为后续优化提供了清晰的路径。以下按优化所需改动的影响范围分类讨论：

#### 底层可直接优化的热点

- **格点稀疏性（non0tab）**：当前程序使用稠密 $\varphi_{\mu g}$ 存储。引入 PySCF 风格的 non0tab 稀疏掩码需要在 `NIMatmul` 中增加掩码字段、在 `prepare_ao` 中生成掩码、在纯函数层增加掩码参数或新的纯函数。这影响中间层和纯函数层，但不影响 Fock 接口层。

- **Psi4 风格 blocking**：将 $\varphi_{\mu g}$ 压缩为 $\varphi_{\mu' g}^{\text{packed}}$ 加映射表，需要在中间层引入新的数据结构和纯函数层增加对应的缩并函数。同样不影响 Fock 接口层。

- **`contract_ao_wv` 系列函数**是计算瓶颈。当前使用标准 GEMM（Faer 后端）。考虑到 DFT 格点与基组存在稀疏性，且格点数通常远大于原子轨道数；未来
  - 引入 Psi4 形式的 AO packing (目前 REST 已经采用类似策略)；
  - 可以开发专门针对格点-基组乘积的 micro-kernel，以更好地利用缓存和 SIMD 指令。对于小型体系，GEMM 已经非常高效；但对于大型体系，定制化的 kernel 可能带来显著性能提升 (参考 github:ajz34/cfgable-matmul)。

#### 需要多层协同的优化

- **bra-ket 低秩优化**：当前已支持 `homogeneous_braket` 和 `one_bra_mult_ket` 等形式。对于 fxc 的 bra-trans 变种（`make_fxc_pot_with_eff_bra_trans`），输出从 `[nao, nao]` 变为 `[nao, nocc]`，这对上层算法（如 TDDFT 求解器）的数据结构有影响。因此，bra-trans 的推广需要 Fock 接口层和调用方共同适配。

- **格点分批策略**：`nbatch` 的选择影响内存用量与并行效率。对于特别大的体系，可能需要更精细的分批策略（如按原子分批而非按格点顺序分批），这需要在中间层调整 `NIMatmul` 的分批逻辑。

- **泛函计算并行**：`libxc_eval_eff_parallel` 的默认分块大小依密度类型调整。如果嵌套并行（Fock 接口层已在 rayon 线程池内），需要退化为串行。未来可能需要更灵活的并行策略（如独立线程池）。

#### 暂不考虑的优化

- **$\mu, \nu$ 对称性利用（SYRK 类优化）**：$\rho_g$ 和 $\tau_g$ 的计算可以利用密度矩阵的对称性减少约一半计算量，但 $\rho_g^t$ 不行。考虑到 GGA/mGGA 是主流计算任务，且 SYRK 的 micro-kernel 与 GEMM 不同，暂不优先考虑。

- **复数类型支持**：当前仅支持 `f64`。复数类型需要新的 micro-kernel 和共轭关系处理，暂不实现。

- **LAPL 型 mGGA 的势缩并**：当前密度格点可以计算 LAPL 分量，但 `contract_ao_wv` 与 LibXC 目前不支持 LAPL 缩并（需要 AO 二阶导数，增加 GEMM 数量且应用场景有限）。


## 3. 设计细节或决定

### 3.1 为何使用张量库 (rstsr)

本程序使用 rstsr 作为张量库，而非直接操作原始数组或依赖 BLAS/LAPACK。理由如下：

- rstsr 是 Rust 下的张量运算库（类似 NumPy），支持任意维度张量、列优先/行优先两种布局，后端使用 Faer 实现矩阵运算。
- 使用 rstsr 可以避免直接依赖 BLAS（Faer 是纯 Rust 实现的线性代数库），编译更方便。
- rstsr 的 `i()` 切片语法（类似 NumPy 的索引）使得多维张量的子视图操作非常简洁，这对 DFT 格点积分中频繁出现的 `[ngrids, nao, ncomp]` 三维张量的切片操作、以及更高维度的开壳层 `fxc_eff`/`kxc_eff` 尤为重要。
- rstsr 的运算与 NumPy 运算基本一一对应。这使得我们可以非常方便地将 NumPy 参考实现中的张量操作直接翻译为 rstsr 代码，降低实现难度并减少出错概率。

### 3.2 列优先 (column-major)

这是由于 REST 项目采用列优先。

实际上，大多数计算化学程序确实是采用列优先的。目前计算化学惯用语言是 Fortran/C++；其中 Fortran 默认 column-major，而 C++ 许多程序采用的 Arma 框架也是 column-major。PySCF 作为 Python 的计算化学库，虽然 Python 本身是 row-major，但 PySCF 内部的 C 代码有一些是 column-major 的；其 Python 接口，特别是 DFT 部分，经常传递的是 F-contiguous 或混合连续的 NumPy 高维向量。

### 3.3 密度矩阵的对称性与 contract_ao_wv 的系数

`contract_ao_wv_without_symmetrize` 先计算非对称半矩阵，再通过 $V += V^T$ 对称化。这一策略导致不同密度类型下系数的差异：

- LDA: 系数 0.5。对称化后 $V_{\mu\nu} = 0.5 \times (\text{half}) + 0.5 \times (\text{half})^T = \text{full}$。
- GGA: 系数 1.0。因为 GGA 贡献本身是 $\varphi_{\mu g}^t (w_g f^{\rho_t} \varphi_{\nu g}) + \text{swap}(\mu,\nu)$，两项恰好对应非对称半矩阵与其转置。
- mGGA (tau): 系数 0.25。对称化后得到 $0.25 \times (\text{half}) + 0.25 \times (\text{half})^T = 0.5 \times \text{full}$，与 $\frac{1}{2}\sum_t w_g f^\tau \varphi_{\mu g}^t \varphi_{\nu g}^t$ 一致。

这一策略的优点是：所有密度类型的贡献可以统一地累加到同一个非对称半矩阵中，最后一步对称化即可。缺点是：对于 LDA 和 mGGA，由于 $\mu, \nu$ 对称性本可减少一半计算量，当前策略实际上计算了完整的非对称矩阵再对称化，浪费了一半的 GEMM 计算量。但考虑到 GGA 是主流任务（其 $\rho_g^t$ 项不能利用对称性），且统一策略简化了程序实现，暂不引入 SYRK 类优化。

### 3.4 对负占据数没有支持

bra-ket 形式中，bra 通常构造为 $C_{\mu i} \sqrt{n_i}$（$n_i$ 为占据数）。这要求占据数 $n_i \geq 0$。对于大多数自洽场方法，占据数为正或零；但对于某些特殊方法（如 fractional occupation 或某些 DFT 对稳定性分析的处理），可能出现负占据数。负占据数出现的情形非常少，当前程序不支持负占据数。

如果希望在计算负占据数密度矩阵对应的密度格点，或者直接传入密度 (而不传入轨道)，或者作两次计算：分别得到正占据密度、负的负占据密度，然后两者相减。

### 3.5 使用 $\rho_t$ 而非 $\gamma$ 作为泛函基本变量

本程序在泛函计算中，使用密度梯度分量 $\rho_g^t$ ($t \in \{x,y,z}$) 而非 $\gamma = |\nabla\rho|^2$ 作为泛函导数的基本变量。理由已在 2.2 节步骤二中说明：$\nabla\rho$ 是密度矩阵的一阶量，而 $\gamma$ 是二阶量。使用 $\rho_t$ 使得后续程序推导更为简单。

代价是格点维度从 LDA/GGA/mGGA 的 1/2/3 增加到 1/4/5（自旋非极化），但这是在 eval_xc 层面的增加，不反应在 GEMM 瓶颈中。

在程序实现中，sigma unfolding（从 LibXC 的 $\gamma$ 导数到 $\rho_t$ 导数的变换）在 `xc_deriv.rs` 中通过链式法则 $f^{\rho_t} = 2 f^\gamma \rho_t$ 实现。对于二阶和三阶导数，还需要处理 $\partial^2/\partial\gamma^2$ 的对角修正项。

### 3.6 LAPL 型 mGGA 的限制

当前程序支持 LAPL 密度格点的计算（$\nabla^2\rho = 4\tau + 2\sum_{\mu\nu} \varphi_{\mu g} D_{\mu\nu} (\varphi_{\nu g,xx} + \varphi_{\nu g,yy} + \varphi_{\nu g,zz})$），但 `contract_ao_wv` 不支持 LAPL 缩并。这是因为 LAPL 缩并需要 AO 二阶导数（$n_\text{comp} = 10$），增加 GEMM 数量，且 LAPL 型泛函的应用场景有限。

### 3.7 NIDenType 枚举的设计

`NIDenType` 枚举（RHO/SIGMA/TAU/LAPL）采用递进式设计：每个高级类型包含所有低级类型的分量。这使得 `NIDenType` 可以同时控制：

- 输出密度格点的分量数 `num_nvar()` (1/4/5/6)
- 所需 AO 导数阶 `num_ao_deriv()` (0/1/1/2)
- 所需 AO 分量数 `num_ao_comp()` (1/4/4/10)

密度分量顺序统一为 $\rho, \rho_x, \rho_y, \rho_z, \tau, \nabla^2\rho$，与 `NIDenType` 的递进关系一致。无论泛函是否需要 $\tau$，LAPL 类型中 $\tau$ 始终是第 5 个分量（而非第 4 个），这保证了分量索引的一致性。

### 3.8 格点分批 (nbatch) vs 格点分块 (nchunk)

`NIMatmul` 中有两个粒度参数：

- `nbatch`：内存控制参数。完整 AO 张量 `[ngrids, nao, ncomp]` 可能过大（例如 1000 原子体系可达数 GB），因此按 `nbatch` 分批处理。每批独立计算 AO、密度、泛函和势矩阵，累积结果。`nbatch` 默认为 $384 \times 4 \times n_\text{threads}$。
- `nchunk`：并行粒度参数。在纯函数层中，格点按 `nchunk` 分块分配给不同线程。`nchunk` 应对应 GEMM 的 KC 维度（通常 256-512），以获得较好的缓存利用率。默认为 384。

两者关系为 full-grid > batch > chunk > per-grid = 1。`nbatch` 应为 `nchunk` 的倍数以获得更好的性能。

### 3.9 fxc/kxc 有效势的格点空间缩并

对于二阶 (fxc) 和三阶 (kxc) 响应，泛函输出是高维张量（$f^{\xi\xi'}$ 或 $f^{\xi\xi'\xi''}$），不能直接传入 `contract_ao_wv`。需要先在格点空间作缩并：

- fxc：$\text{fxc\_eff\_contracted}_\xi = \sum_{\xi'} f^{\xi\xi'} \xi'_g[\mathbf{R}]$，得到 `[ngrids, nvar]` 的有效势
- kxc：$\text{kxc\_eff\_contracted}_\xi = \sum_{\xi'\xi''} f^{\xi\xi'\xi''} \xi'_g[\mathbf{R}'] \xi''_g[\mathbf{R}'']$，得到 `[ngrids, nvar]` 的有效势

缩并后的有效势维度与 vxc_eff 相同，因此可以复用 `contract_ao_wv` 函数。这一设计使得 `contract_ao_wv` 的接口对所有导数阶保持一致。

### 3.10 UKS 的自旋维度约定

开壳层 (UKS) 的张量维度中有一些约定：

- rho: `[ngrids, nvar, 2]`（而非 `[ngrids, 2, nvar]`）
- fxc_eff: `[ngrids, nvar, 2, nvar, 2]`
- fxc 输出: `[nao, nao, 2, nset]` (而非 `[nao, nao, nset, 2]`)，这与 PySCF 有区别。

### 3.11 LDA 维度约定

LDA 的 $n_{\text{var}} = 1$，即只有一个密度分量。PySCF 中经常将该分量约去 (squeeze)，使得 rho (闭壳层下) 的维度为 `[ngrids]` 而非 `[ngrids, 1]`。

但我们为了尽可能统一接口，保持所有密度类型的维度结构一致，选择不约去 LDA 的分量，使得闭壳层下 LDA 的 rho 维度为 `[ngrids, 1]`。
