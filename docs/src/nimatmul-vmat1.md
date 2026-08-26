# DFT Fock 矩阵一阶 Skeleton 导数

这份文档将讨论 DFT Fock 矩阵一阶 Skeleton 导数的计算公式与实现。我们只使用矩阵乘法的策略，不使用其他优化工具。

> **该文档仅处理闭壳层问题**
>
> 对于开壳层，一部分计算过程需要依自旋重新处理。我们会在其他文档中讨论该问题。

> **该文档有 AI 参与编写**
>
> 3.2 节的内容是由 AI 生成 (Claude Code + glm-5.3)。有一定的人工审核。

## 1. 一阶 Fock 矩阵 Skeleton 导数：概论

我们首先回顾 Fock 矩阵的定义。Fock 矩阵是能量对密度 (作为变分参数) 的一阶导数：

$$
V_{\mu \nu}^\text{xc} = \frac{\partial E^\text{xc}}{\partial D_{\mu \nu}}
$$

出于一些实际考量，这里的导数可能因为闭壳层会引入 2 倍缩放系数。上式对所有能量贡献项都成立；当然我们这里仅讨论 DFT 格点积分。

代入 DFT 的能量计算表达式

$$
E^\text{xc} = \sum_g w_g (f \rho)_g
$$

可以得到 Fock 矩阵的格点积分表示：

$$
V_{\mu \nu}^\text{xc}
= \sum_{\chi g} w_g \frac{\partial (f \rho)_g}{\partial \xi_g^\chi} \frac{\partial \xi_g^\chi}{\partial D_{\mu \nu}}
= \sum_{\chi g} w_g f_g^\chi \xi_{g \mu \nu}^\chi
$$

现在我们要再进行一次导数，但并非是密度矩阵的导数，而是对 Fock 矩阵求原子坐标的导数。我们注意到与原子坐标有显式关系的项包含两部分：泛函核 $f_g^\chi$ 与 DFT 基本参量在基组下的表示 $\xi_{g \mu \nu}^\chi$。

$$
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} = \sum_{\chi g} w_g \frac{\partial f_g^\chi}{\partial A_t} \xi_{g \mu \nu}^\chi + \sum_{\chi g} w_g f_g^\chi \frac{\partial \xi_{g \mu \nu}^\chi}{\partial A_t} \quad (\texttt{vmat\_deriv1})
$$

上式中前者将会产生泛函核二阶导数 $f_{g}^{\chi \chi'}$，因而记为 `vmat_fxc`；后者仅涉及泛函核一阶导数 $f_g^\chi$，因而记为 `vmat_vxc`。我们将分别讨论这两部分的计算公式与实现。

> **缺失格点偏置导数**
> 
> 上面的讨论在理想的全空间坐标积分下成立。现实里，在不少情况下也确实可以给出合理的 Hessian (指杂化 LDA/GGA 泛函)。
> 
> 但作为数值积分方法，我们没有考虑原子坐标变化对格点 $\bm{r}_g$ 与权重 $w_g$ 的影响。meta-GGA 的 Hessian 计算一般要求引入格点偏置的影响。我们将在其他文档中讨论该问题。

## 2. `fxc` 部分实现细节

### 2.1 `fxc` 贡献项实现概述

`fxc` 贡献项的核心问题是

$$
\begin{align*}
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} &\leftarrow \sum_{\chi g} w_g \frac{\partial f_g^\chi}{\partial A_t} \xi_{g \mu \nu}^\chi
\quad (\texttt{vmat\_fxc}) \\
&= \sum_{\chi \chi' g} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^{\chi'}}{\partial A_t} \xi_{g \mu \nu}^\chi
\end{align*}
$$

我们已经在能量 Skeleton 二阶导数中，计算了密度格点对坐标的导数量 `drho` $\partial_{A_t} \xi_g^{\chi'}$，因此上式的化简就是最终的形式。

**函数 `get_vmat_fxc`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `xc_type` | | | `LDA` / `GGA` / `MGGA` | |
| `ao` | $\phi_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>1/4/4 |
| `drho` | $\partial_{A_t} \xi_g^{\chi'}$ | $(g, \chi', t, A)$</br>`[g, x, t, A]` | `[ngrids, nvar, 3, natm]` | |
| `wf` | $w_g f_g^{\chi \chi'}$ | $(g, \chi, \chi')$</br>`[g, x, y]` | `[ngrids, nvar, nvar]` | $\mathrm{sym} (\chi, \chi')$ |
| `vmat_fxc`</br>(output) | | $(\mu, \nu, t, A)$</br>`[u, v, t, A]` | `[nao, nao, 3, natm]` | $\mathrm{sym} (\mu, \nu)$ |

### 2.2 `fxc` 贡献项的具体实现

具体实现上，首先注意到权重 $w_g$、泛函核二阶导数 $f_g^{\chi \chi'}$ 与密度格点对坐标的导数 $\partial_{A_t} \xi_g^{\chi'}$ 都没有涉及到原子轨道 $\mu, \nu$。这些量可以先进行代价较小的、对 $\chi'$ 指标的缩并，得到四维中间量：

$$
\mathscr{T}_{g \chi}^{A_t} = \sum_{\chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^{\chi'}}{\partial A_t}
\quad (\texttt{wf\_rho})
$$

在实际实现中，**我们采用外部依原子指标 `A` 与三维坐标 $t$ 迭代的策略，将当前问题转化为完全等价于普通 Fock 矩阵生成的问题**；换句话说，区别仅仅是我们要生成 $3 n_\mathrm{atom}$ 个矩阵，但每个矩阵在程序实现上与 Fock 矩阵没有区别。

**`vmat_fxc`: LDA (RHO)**

先作数乘计算得到 `aow`；由于在指标 `A, t` 循环内部，`aow` 会是 2-dim 张量 $(g, \mu)$。同时留意 $\chi$ 只有一个分量 $\rho$，因此可以省略 $\chi$ 指标。

$$
\mathscr{W}_{g \mu}^{A_t} = \mathscr{T}_{g}^{A_t} \phi_{g \mu} \quad (\texttt{aow})
$$

随后就可以直接进行矩阵乘法，得到 Fock 矩阵对原子坐标的导数：

$$
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} \leftarrow \sum_{g} \mathscr{W}_{g \mu}^{A_t} \phi_{g \nu} \quad (\texttt{vmat\_fxc}, \text{LDA})
$$

仅仅是出于后续程序的实现方便，这里引入关于 $(\mu, \nu)$ 的对称化，因此会有 0.5 倍系数：

$$
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} \leftarrow \frac{1}{2} \sum_{g} \mathscr{W}_{g \mu}^{A_t} \phi_{g \nu} + \mathrm{swap} (\mu, \nu) \quad (\texttt{vmat\_fxc}, \text{LDA})
$$

由于计算过程中 memory footprint 较小的张量是 `wf_rho` $\mathscr{T}_{g}^{A_t}$，因此一个技巧是直接在 `wf_rho` 上作系数缩放，而不是在计算过程中或结果处缩放。

**`vmat_fxc`: GGA (SIGMA) / MGGA (TAU)**

首先我们给出正常的推演：

$$
\begin{align*}
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} 
&\leftarrow \sum_{g} \mathscr{T}_{g, \chi = \rho}^{A_t} \phi_{g \mu} \phi_{g \nu} \quad (\text{RHO}) \\
&\quad + \sum_{g} \sum_{r \in \{x, y, z\}} \mathscr{T}_{g, \chi = \rho^r}^{A_t} (\phi_{g \mu}^r \phi_{g \nu} + \phi_{g \mu} \phi_{g \nu}^r) \quad (\text{SIGMA}) \\
&\quad + \sum_{g} \sum_{r \in \{x, y, z\}} \mathscr{T}_{g, \chi = \tau}^{A_t} \cdot \frac{1}{2} \phi_{g \mu}^r \phi_{g \nu}^r \quad (\text{TAU})
\end{align*}
$$

我们需要利用一个巧合：密度分量 $\chi \in \{ \rho, \rho^x, \rho^y, \rho^z \}$ (`nvar = 4`) 与原子轨道导数分量 $* \in \{ 1, x, y, z \}$ (`ncomp = 4`) 之间存在对应关系。基于这个巧合，同时考虑到即将引入的 $\mathrm{swap} (\mu, \nu)$ 对称化，我们首先重新确定 `wf_rho` 的缩放系数：

$$
\bar{\mathscr{T}}_{g, \chi}^{A_t} = \begin{cases}
\frac{1}{2} \mathscr{T}_{g, \chi}^{A_t} & \chi = \rho \\
\mathscr{T}_{g, \chi}^{A_t} & \chi = \rho^r, r \in \{ x, y, z \} \\
\frac{1}{4} \mathscr{T}_{g, \chi}^{A_t} & \chi = \tau 
\end{cases}
$$

**对于 GGA 的情况**，我们就充分利用这个巧合，预先缩并指标 $\chi \in \{ \rho, \rho^x, \rho^y, \rho^z \}$ 与对应的 $* \in \{ 1, x, y, z \}$，构造 2-dim 临时张量 `aow` $\mathscr{W}_{g \mu}^{A_t}$，其维度为 $(g, \mu)$：

$$
\mathscr{W}_{g \mu}^{A_t} = \sum_{\substack{\chi \in \{ \rho, \rho^x, \rho^y, \rho^z \} \\ * \in \{ 1, x, y, z \}}} \bar{\mathscr{T}}_{g \chi}^{A_t} \phi_{g \mu}^* \quad (\texttt{aow})
$$

随后作简单的矩阵乘法：

$$
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} \leftarrow \sum_{g} \mathscr{W}_{g \mu}^{A_t} \phi_{g \nu} + \mathrm{swap} (\mu, \nu) \quad (\texttt{vmat\_fxc}, \text{GGA})
$$

**对于 MGGA 增量的情况**，我们需要对 $r \in \{ x, y, z \}$ 三种情况作循环，分别都构造一次临时张量 `aow` 张量 $\mathscr{W}_{g \mu}^{A_t r}$，其维度为 $(g, \mu)$：

$$
\mathscr{W}_{g \mu}^{A_t r} = \bar{\mathscr{T}}_{g, \chi = \tau}^{A_t} \phi_{g \mu}^r \quad (\texttt{aow})
$$

随后作增量矩阵乘法：

$$
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} \leftarrow \sum_{g} \sum_{r \in \{x, y, z\}} \mathscr{W}_{g \mu}^{A_t r} \phi_{g \nu}^r + \mathrm{swap} (\mu, \nu) \quad (\texttt{vmat\_fxc}, \text{MGGA increment})
$$

## 3. `vxc` 部分实现细节

### 3.1 `vxc` 贡献项实现概述

`vxc` 贡献项的核心问题是

$$
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} \leftarrow \sum_{\chi g} w_g f_g^\chi \frac{\partial \xi_{g \mu \nu}^\chi}{\partial A_t} \quad (\texttt{vmat\_vxc})
$$

在这一项的处理上，我们对 $\delta_{\mu \in A}$ 的应用，会在最后进行。我们先会产生一个类似于对电子坐标作导数的中间矩阵 `vmat_ip`。这与 Skeleton 能量二阶导数的 `dao_vxc_diag/off` 的情况是类似的。

我们首先需要将原子核导数 $\partial_{A_t}$ 问题转化为电子坐标导数 $\partial_{t}$ 问题 (借用 $\partial_{A_t} \phi_{g \mu}^* = - \partial_t \phi_{g \mu}^* \delta_{\mu \in A}$ 的关系)。为了后续处理方便，我们定义细节更丰富的、仅针对基组指标 $\mu$ 作偏导的临时张量 $\mathscr{T}_{\mu \nu}^{t}$。需要留意，该张量 **并非对 $\mu, \nu$ 对称**。

$$
\mathscr{T}_{\mu \nu}^{t} = \sum_{\chi g} w_g f_g^\chi \frac{\partial \xi_{g \mu \nu}^\chi}{\partial t} \quad (\text{restrict $\partial$ to $\mu$}, \texttt{vmat\_ip})
$$

随后再依原子指标 $A$ 作缩并，得到最终的 vxc 贡献项：

$$
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t} = - \mathscr{T}_{\mu \nu}^{t} \delta_{\mu \in A} + \mathrm{swap} (\mu, \nu)
$$

**函数 `get_vmat_ip`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `xc_type` | | | `LDA` / `GGA` / `MGGA` | |
| `ao` | $\phi_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>4/10/10 |
| `wv` | $w_g f_g^\chi$ | $(g, \chi)$</br>`[g, x]` | `[ngrids, nvar]` | |
| `vmat_ip`</br>(output) | $\mathscr{T}_{\mu \nu}^{t}$ | $(\mu, \nu, t)$</br>`[u, v, t]` | `[nao, nao, 3]` | |

**函数 `get_vmat_vxc`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `vmat_ip` | $\mathscr{T}_{\mu \nu}^{t}$ | $(\mu, \nu, t)$</br>`[u, v, t]` | `[nao, nao, 3]` | |
| `aoslices` | | | `natm` | |
| `vmat_vxc`</br>(output) | | $(\mu, \nu, t, A)$</br>`[u, v, t, A]` | `[nao, nao, 3, natm]` | $\mathrm{sym} (\mu, \nu)$ |

### 3.2 `vxc` 贡献项的具体实现

`vxc` 贡献项的实现结构与 Fock 矩阵实际上是类似的，只是需要留意与 `fxc` 的情形有两个不同：
- `vxc` 贡献项的中间量 $\mathscr{T}_{\mu \nu}^{t}$ 并非对 $(\mu, \nu)$ 对称，在编写代码的时候需要稍微谨慎一些。
- GGA (SIGMA) 所构造的 $\xi_{g \mu \nu}^{\rho^r}$ ($r \in \{x, y, z\}$) 中，左矢 $\phi_{g \mu}^r$ 与右矢 $\phi_{g \nu}$ 的导数分量不对称，因此会相对于 Fock 矩阵计算多出一些额外的代码实现。

`vmat_ip` 对原子指标没有任何依赖，只需一次性生成 $t \in \{x, y, z\}$ 的 3 个矩阵；原子指标 `A` 的依赖完全推迟到 `get_vmat_vxc` 中对 $\delta_{\mu \in A}$ 的缩并过程里。

**`vmat_ip`: LDA (RHO)**

LDA 部分对应 $\chi = \rho$，$\xi_{g \mu \nu}^{\chi = \rho} = \phi_{g \mu} \phi_{g \nu}$。仅对 $\mu$ 作偏导，

$$
\frac{\partial \xi_{g \mu \nu}^{\chi = \rho}}{\partial t} = \phi_{g \mu}^t \phi_{g \nu} \quad \text{(restrict $\partial$ to $\mu$)}
$$

由于仅对 $\mu$ 作偏导，导数指标 $t$ 固定在左矢上，因此数乘加权作用在右矢上；先作数乘计算得到 2-dim 临时张量 `aow` $\mathscr{W}_{g \nu}$ (留意其携带的是右矢指标 $\nu$，这与 `fxc` 的 `aow` $\mathscr{W}_{g \mu}$ 不同；对于 LDA，加权作用在哪一侧所得矩阵相同，选择右矢是为了与 GGA 的第一个缩并结构保持一致)：

$$
\mathscr{W}_{g \nu} = w_g f_g^{\rho} \phi_{g \nu} \quad (\texttt{aow})
$$

随后对每个 $t$ 作矩阵乘法：

$$
\mathscr{T}_{\mu \nu}^t = \sum_{g} \phi_{g \mu}^t \mathscr{W}_{g \nu} \quad (\texttt{vmat\_ip}, \text{LDA})
$$

**`vmat_ip`: GGA (SIGMA)**

GGA 任务对应 RHO + SIGMA 两部分的叠加。仅对 $\mu$ 作偏导，

$$
\frac{\partial \xi_{g \mu \nu}^{\chi = \rho}}{\partial t} = \phi_{g \mu}^t \phi_{g \nu} \quad (\text{RHO}), \qquad \frac{\partial \xi_{g \mu \nu}^{\chi = \rho^r}}{\partial t} = \phi_{g \mu}^t \phi_{g \nu}^r + \phi_{g \mu}^{t r} \phi_{g \nu} \quad (\text{SIGMA})
$$

代入 $\mathscr{T}_{\mu \nu}^t$ 的定义：

$$
\begin{align*}
\mathscr{T}_{\mu \nu}^t
&= \sum_{g} w_g f_g^{\rho} \phi_{g \mu}^t \phi_{g \nu} \quad (\text{RHO}) \\
&\quad + \sum_{g} \sum_{r \in \{x, y, z\}} w_g f_g^{\rho^r} \left( \phi_{g \mu}^t \phi_{g \nu}^r + \phi_{g \mu}^{t r} \phi_{g \nu} \right) \quad (\text{SIGMA})
\end{align*}
$$

SIGMA 的两个求和项中，第一项 $\phi_{g \mu}^t \phi_{g \nu}^r$ 的两个导数分量分居左矢与右矢，第二项 $\phi_{g \mu}^{t r} \phi_{g \nu}$ 的两个导数分量都在左矢上；因此 `fxc` 中 $\chi \leftrightarrow *$ 的完全一致的重合，这里并不能直接应用，无法像 2.2 节那样用单个 `aow` 张量完成缩并。但上式仍可以拆分为两个矩阵乘法结构。

第一个结构将数乘加权作用在右矢上。临时张量 `aow` $\mathscr{W}_{g \nu}$ 不依赖指标 $t$，可提出到 $t$ 循环外：

$$
\mathscr{W}_{g \nu} = \frac{1}{2} w_g f_g^{\rho} \phi_{g \nu} + \sum_{r \in \{x, y, z\}} w_g f_g^{\rho^r} \phi_{g \nu}^r \quad (\texttt{aow})
$$

$$
\mathscr{T}_{\mu \nu}^t \mathrel{+}= \sum_{g} \phi_{g \mu}^t \mathscr{W}_{g \nu}
$$

该结构覆盖 SIGMA 第一项 $\phi_{g \mu}^t \phi_{g \nu}^r$，以及 RHO 项的一半。

第二个结构将数乘加权作用在左矢上。临时张量 `aow_d` $\mathscr{W}_{g \mu}^t$ 依赖指标 $t$，其中 $\phi_{g \mu}^{t r}$ 取自 `ao` 的二阶导数分量：

$$
\mathscr{W}_{g \mu}^t = \frac{1}{2} w_g f_g^{\rho} \phi_{g \mu}^t + \sum_{r \in \{x, y, z\}} w_g f_g^{\rho^r} \phi_{g \mu}^{t r} \quad (\texttt{aow\_d})
$$

$$
\mathscr{T}_{\mu \nu}^t \mathrel{+}= \sum_{g} \mathscr{W}_{g \mu}^t \phi_{g \nu} \quad (\texttt{vmat\_ip}, \text{GGA})
$$

该结构覆盖 SIGMA 第二项 $\phi_{g \mu}^{t r} \phi_{g \nu}$，以及 RHO 项的另一半。

RHO 项 $\phi_{g \mu}^t \phi_{g \nu}$ 的指标形状与两个结构都兼容，因此被平分为两个 $\frac{1}{2}$ 系数分别并入两结构，相加后总系数恢复为 1。需要留意，此处 $\frac{1}{2}$ 的动机与 `fxc` (2.2 节) 完全不同：`fxc` 的缩放系数 $\bar{\mathscr{T}}$ 服务于 $\mathrm{swap} (\mu, \nu)$ 对称化；而 `vmat_ip` 在函数内部不作对称化，$\frac{1}{2}$ 纯粹是为了将 RHO 项并入上述两个矩阵乘法结构，避免单独为 RHO 项再作一次矩阵乘法。

程序上，本情况对 `ao` 的读取范围最高到二阶导数分量 ($\phi$、$\phi^r$、$\phi^{t r}$，即 3.1 节表格中的 `ncomp = 10`)。计算量为 $6 \times 2 n_\mathrm{basis}^2 n_\mathrm{grid}$ FLOPs，即两个结构各依 $t$ 作 3 次矩阵乘法。

**`vmat_ip`: MGGA (TAU) 增量**

MGGA 增量对应 $\chi = \tau$，$\xi_{g \mu \nu}^{\chi = \tau} = \frac{1}{2} \sum_{r \in \{x, y, z\}} \phi_{g \mu}^r \phi_{g \nu}^r$。仅对 $\mu$ 作偏导，

$$
\frac{\partial \xi_{g \mu \nu}^{\chi = \tau}}{\partial t} = \frac{1}{2} \sum_{r \in \{x, y, z\}} \phi_{g \mu}^{t r} \phi_{g \nu}^r \quad \text{(restrict $\partial$ to $\mu$)}
$$

我们需要对 $r \in \{x, y, z\}$ 三种情况作循环，分别都构造一次 2-dim 临时张量 `aow` $\mathscr{W}_{g \nu}^r$：

$$
\mathscr{W}_{g \nu}^r = \frac{1}{2} w_g f_g^{\tau} \phi_{g \nu}^r \quad (\texttt{aow})
$$

随后作增量矩阵乘法：

$$
\mathscr{T}_{\mu \nu}^t \mathrel{+}= \sum_{g} \phi_{g \mu}^{t r} \mathscr{W}_{g \nu}^r \quad (\texttt{vmat\_ip}, \text{MGGA increment})
$$

此处的 $\frac{1}{2}$ 并非实现技巧，而是 $\xi_{g \mu \nu}^{\chi = \tau}$ 定义中固有的系数；对比 `fxc` 的 MGGA 增量，那里由于附加的 $\mathrm{swap} (\mu, \nu)$ 对称化而取 $\frac{1}{4}$ (固有 $\frac{1}{2}$ 与对称化 $\frac{1}{2}$ 之积)。循环次序取外层 $r$、内层 $t$，使 `aow` $\mathscr{W}_{g \nu}^r$ 仅依赖 $r$ 而在内层循环中复用。计算量为额外的 $9 \times 2 n_\mathrm{basis}^2 n_\mathrm{grid}$ FLOPs，即依 $r \times t$ 共 $3 \times 3 = 9$ 次矩阵乘法。

