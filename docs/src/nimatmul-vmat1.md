# DFT Fock 矩阵一阶 Skeleton 导数

这份文档将讨论 DFT Fock 矩阵一阶 Skeleton 导数的计算公式与实现。我们只使用矩阵乘法的策略，不使用其他优化工具。

> **该文档仅处理闭壳层问题**
>
> 对于开壳层，一部分计算过程需要依自旋重新处理。我们会在其他文档中讨论该问题。

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
| `wf` | $w_g f_g^{\chi \chi'}$ | $(g, \chi, \chi')$</br>`[g, x, y]` | `[ngrids, nvar, nvar]` | $\mathrm{sym} (\chi, \chi'
)$ |
| `aoslices` | | | `natm` | 仅用于 `natm` |
| `vmat_fxc`</br>(output) | | $(\mu, \nu, t, A)$</br>`[u, v, t, A]` | `[nao, nao, 3, natm]` | $\mathrm{sym} (\mu, \nu)$ |

具体实现上，首先注意到权重 $w_g$、泛函核二阶导数 $f_g^{\chi \chi'}$ 与密度格点对坐标的导数 $\partial_{A_t} \xi_g^{\chi'}$ 都没有涉及到原子轨道 $\mu, \nu$。这些量可以先进行代价较小的、对 $\chi'$ 指标的缩并，得到四维中间量：

$$
\mathscr{T}_{g \chi}^{A_t} = \sum_{\chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^{\chi'}}{\partial A_t}
\quad (\texttt{wf\_rho})
$$

在实际实现中，**我们采用外部依原子指标 `A` 与三维坐标 $t$ 迭代的策略，将当前问题转化为完全等价与普通 Fock 矩阵生成的问题**；换句话说，区别仅仅是我们要生成 $3 n_\mathrm{atom}$ 个矩阵，但每个矩阵在程序实现上与 Fock 矩阵没有区别。

这里作为参考，将详细的缩并过程列出。

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

由于计算过程中 memory footprint 较小的张量是 `wf_rho` $\mathscr{T}_{g}^{\chi A_t}$，因此一个技巧是直接在 `wf_rho` 上作系数缩放，而不是在计算过程中或结果处缩放。

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
