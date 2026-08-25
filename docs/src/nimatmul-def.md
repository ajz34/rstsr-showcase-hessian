# DFT 格点积分约定

这份文档的主要目的是，对程序中使用的 DFT 格点积分约定进行说明。这包括公式符号、常用程序变量、维度信息等等。

该文档仅对 nimatmul，一种简易 DFT 格点积分模块，的约定进行说明。

## 1. DFT 能量

这一小节以最快的方式，说明 DFT 格点积分的核心目标。即使 nimatmul 是非常 naive 的 DFT 格点积分实现，公式与程序实现会有大量繁杂的细节；但所有工作都无外乎是 DFT 能量与其导数计算。

DFT 格点积分是在泛函核 $f(\rho, \gamma, \tau, \cdots)$ 下，对空间进行数值积分的过程：

$$
E^\text{xc} = \int f \cdot \rho (\bm{r}) \, \mathrm{d} \bm{r}
$$

通常的 DFT 程序是格点离散的。格点离散是一种近似，但我们始终以等号表述。

$$
E^\text{xc} = \sum_g w_g f_g \rho_g
$$

其中，
- $g$ 是格点索引，
- $\rho_g$ 是格点 $g$ 处的电子密度 $\rho(\bm{r}_g)$；
- $f_g$ 是格点 $g$ 处的泛函核 $f(\rho_g, \gamma_g, \tau_g, \cdots)$；
- $w_g$ 是格点 $g$ 处的权重。 

> 补充约定：我们暂时不讨论 Laplacian 型泛函。因此，最终我们的 DFT 基本参量仅包含 $\rho$、$\gamma$、$\tau$，而不包含 $\nabla^2 \rho$。
>
> 补充说明：许多程序会使用 sigma 表示 GGA 的 $\gamma = \nabla \rho \cdot \nabla \rho$。由于我们使用该记号作为自旋分量的索引，因此我们不使用 sigma 表示 GGA 的梯度密度。

## 2. DFT 基本参量 $\xi$

为了以后程序实现与公式推演方便，记 DFT 基本参量为 $\bm{\xi}$ 的向量形式[^note-1]：

[^note-1]: 该记号是我个人的习惯。该记号取材自 Su, N. Q.; Zhang, I. Y.; Xu, X. *J. Comput. Chem.* **2013**, *34* (20), 1759–1774. doi: [10.1002/jcc.23312](https://doi.org/10.1002/jcc.23312)。对应本文的 $\xi_\chi$，该文章的对应记号是 $\zeta_\eta$。之所以要更改 $\zeta$ 到 $\xi$，是因为 $\zeta$ 在 DFT 经常被用来表示自旋极化率 $\zeta = (\rho^\alpha - \rho^\beta) / \rho$，也在基组函数中被用来表示指数衰减参数 $\zeta$。之所以要更改 $\eta$ 到 $\chi$，单纯是因为我们以后可能在 einsum 记号中用 `x` 表示该参量的索引。

$$
\bm{\xi} = (\rho, \rho^x, \rho^y, \rho^z, \tau)
$$

DFT 基本参量 $\bm{\xi}$ 的分量角标记为 $\chi$。请留意对于 LDA, GGA 与 mGGA 的 $\bm{\xi}$ 并不一致。LDA 只有第一个分量 ($\chi$ 长度为 1)；GGA 只有前四个分量 ($\chi$ 长度为 4)；mGGA 则包含全部五个分量 ($\chi$ 长度为 5)。

$$
\xi_{g \mu \nu}^\chi = \begin{cases}
\phi_{g \mu} \phi_{g \nu} & \chi = \rho \\
\phi_{g \mu}^r \phi_{g \nu} + \phi_{g \mu} \phi_{g \nu}^r & \chi = \rho^r \\
\sum_r \frac{1}{2} \phi_{g \mu}^r \phi_{g \nu}^r & \chi = \tau
\end{cases} 
$$

其中，我们以 $r \in \{x, y, z\}$ 表示空间维度索引。表示空间维度还经常用 $t, s$。我们以后会尽量在涉及 GGA 或 mGGA 的密度表达式中使用指标 $r$、而在梯度计算中使用指标 $t, s$ 来表示空间维度索引，以此来区分两者。

需要指出，GGA 传统上是用 $\gamma = \nabla \rho \cdot \nabla \rho$ 作为参量表示的；事实上 DFT 泛函引擎 LibXC 与 XCFun 程序上就是这么设计的。但由于 $\gamma$ 是密度矩阵的二阶量，在程序实现与公式推演时会有相当程度的复杂性。以后的工作中，我们会始终将 $\gamma$ 转换为 $\rho_x, \rho_y, \rho_z$ 的形式来进行处理。

引入密度矩阵 $D_{\mu \nu}$，可以得到 DFT 基本参量的空间格点表示：

$$
\xi_g^\chi = \sum_{\mu \nu} D_{\mu \nu} \xi_{g \mu \nu}^\chi
$$

请留意，在不少情形下，使用轨道系数 $C_{\mu i}$ 替代密度矩阵 $D_{\mu \nu}$ 经常是程序上更效率的做法。这里仅作定义，不详细展开。

## 3. DFT 泛函核导数

我们定义泛函核导数

$$
\begin{aligned}
f^\chi &= \frac{\partial (f \rho)}{\partial \xi^{\chi}} \\
f^{\chi \chi'} &= \frac{\partial^2 (f \rho)}{\partial \xi^{\chi} \partial \xi^{\chi'}} \\
f^{\chi \chi' \chi''} &= \frac{\partial^3 (f \rho)}{\partial \xi^{\chi} \partial \xi^{\chi'} \partial \xi^{\chi''}}
\end{aligned}
$$

请留意，这里被求导的量不是泛函核 $f$ 本身，而是其与密度的乘积 $f \rho$。

## 4. 程序约定

**维度**

| 程序指标 | 公式指标 | 维度大小 | 意义 |
|--|--|--|--|
| `u, v` | 下标 $\mu, \nu$ | `nao` $n_\mathrm{basis}$ | 基函数 |
| `g` | 下标 $g$ | `ngrids` $n_\mathrm{grids}$ | DFT 格点 |
| `t, s, r` | 上标 $t, s, r \in \{x, y, z\}$ | 3 | 空间分量 |
| `x, y` | 上标 $\chi, \chi'$ | `nvar` $n_\mathrm{var}$ | DFT 基本参量分量 |
| `A, B` | 上标 $A, B$ | `natm` $n_\mathrm{atom}$ | 原子 |
| `A, B` | 上标 $\mathbb{A}, \mathbb{B}$ | `nprop` $n_\mathrm{prop}$ | 任意性质 |
| `i, j` | 下标 $i, j$ | `nocc` $n_\mathrm{occ}$ | 占据轨道 |
| | 上标 $*$ | `ncomp` | 原子轨道导数分量 |
| `σ, ς` | 上标 $\sigma, \varsigma \in \{ \alpha, \beta \}$ | 2 | 自旋分量 |

**张量 (闭壳层)**

| 变量名 | 公式表达 | 维度 (PySCF/mixed-major) | 维度 (REST/col-major) |
|--|--|--|--|
| `ao` | $\phi_{g \mu}^{*}$ | $(*, g, \mu)$ <br> `[ncomp, ngrids, nao]` | $(g, \mu, *)$ <br> `[ngrids, nao, ncomp]` |
| `rho` | $\xi_{g}^{\chi}$ | $(\chi, g)$ <br> `[nvar, ngrids]` | $(g, \chi)$ <br> `[ngrids, nvar]` |
| `vxc` | $f_g^\chi$ | $(\chi, g)$ <br> `[nvar, ngrids]` | $(g, \chi)$ <br> `[ngrids, nvar]` |
| `fxc` | $f_g^{\chi \chi'}$ | $(\chi, \chi', g)$ <br> `[nvar, nvar, ngrids]` | $(g, \chi, \chi')$ <br> `[ngrids, nvar, nvar]` |

这里出现的星号 $*$ 泛指原子轨道导数分量；其具体数目依所含导数阶数而定 (连同零阶分量，至一阶导数共 4 个、至二阶共 10 个、至三阶共 20 个)。它并非复共轭。

请留意，PySCF 所使用的 NumPy 尽管是 row-major 约定；但其中的 $\phi_{g \mu}^{*}$ 尽管表面维度是 $(*, g, \mu)$，其内存中的连续性是 $g$ 即格点索引在最内层，其次 $\mu$ 即基函数，最后是原子轨道导数分量。这个内存连续性顺序与 column-major 的 REST 实际上是相同的。对于其他分量，REST (column-major) 与 PySCF/NumPy (row-major) 的维度相反，内存连续性相同。

**张量 (开壳层)**

| 变量名 | 公式表达 | 维度 (PySCF/mixed-major) | 维度 (REST/col-major) |
|--|--|--|--|
| `rho` | $\xi_{g}^{\sigma \chi}$ | $(\sigma, \chi, g)$ <br> `[2, nvar, ngrids]` | $(g, \chi, \sigma)$ <br> `[ngrids, nvar, 2]` |
| `vxc` | $f_g^{\sigma \chi}$ | $(\sigma, \chi, g)$ <br> `[2, nvar, ngrids]` | $(g, \chi, \sigma)$ <br> `[ngrids, nvar, 2]` |
| `fxc` | $f_g^{\sigma \chi \sigma' \chi'}$ | $(\sigma, \chi, \sigma', \chi', g)$ <br> `[2, nvar, 2, nvar, ngrids]` | $(g, \chi, \sigma, \chi', \sigma')$ <br> `[ngrids, nvar, 2, nvar, 2]` |

请留意，`fxc` 张量是具有 $(\sigma, \sigma')$ 与 $(\chi, \chi')$ 对称性的。我们为了程序编写上的方便，放弃了对称性，使用全张量表示。
