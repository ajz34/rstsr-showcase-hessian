# DFT 能量二阶 Skeleton 导数

这份文档将讨论 DFT 二阶 Skeleton 导数的实现策略。我们只使用矩阵乘法的策略，不使用其他优化工具。

> **该文档仅处理闭壳层问题**
>
> 对于开壳层，一部分计算过程需要依自旋重新处理。我们会在其他文档中讨论该问题。

## 1. 二阶 Skeleton 导数：概论

我们将 DFT 二阶 Skeleton 导数分为三部分：
- `fxc` 即 $f^{\chi \chi'}$ 相关部分；
- `vxc_diag` 对角部分 (单原子双重导数)；
- `vxc_off` 非对角部分 (双原子各一重导数)。

我们这里稍作展开。首先，回顾 DFT 能量的计算表达式：

$$
E^\text{xc} = \int f \rho \, \mathrm{d} \bm{r} = \sum_g w_g f_g \rho_g = \sum_g w_g (f \rho)_g
$$

Skeleton 导数的定义是，在密度矩阵 (轨道系数) 不发生变化的情况下，改变原子坐标的导数。Skeleton 导数可以通过固定轨道系数、改变原子坐标，作数值差分计算得到。

对于 DFT 任务，其一阶 Skeleton 导数是 (暂时用 partial 记号表示 Skeleton 导数；我们只要记得不要对轨道系数求导)：

$$
\frac{\partial E^\text{xc}}{\partial \mathbb{A}} = \int \frac{\partial (f \rho)}{\partial \mathbb{A}} \, \mathrm{d} \bm{r} = \int \sum_\chi \frac{\partial (f \rho)}{\partial \xi^\chi} \frac{\partial \xi^\chi}{\partial \mathbb{A}} \, \mathrm{d} \bm{r}
$$

写成格点积分的形式：

$$
\frac{\partial E^\text{xc}}{\partial \mathbb{A}} = \sum_{g \chi} w_g f_g^\chi \frac{\partial \xi_g^\chi}{\partial \mathbb{A}}
$$

对上式再求一次导数，使用乘法的链式法则：

$$
\begin{aligned}
\frac{\partial^2 E^\text{xc}}{\partial \mathbb{A} \partial \mathbb{B}}
&= \sum_{g \chi} w_g \left( \frac{\partial f_g^\chi}{\partial \mathbb{B}} \frac{\partial \xi_g^\chi}{\partial \mathbb{A}} + f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial \mathbb{A} \partial \mathbb{B}} \right) \\
&= \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^{\chi'}}{\partial \mathbb{B}} \frac{\partial \xi_g^\chi}{\partial \mathbb{A}} + \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial \mathbb{A} \partial \mathbb{B}}
\quad (\texttt{de\_xc})
\end{aligned}
$$

到这里，我们已经可以将第一项 (`fxc` 贡献项) 拆分出来了。第二项是 `vxc` 贡献，取决于具体的偏导计算过程，我们将其拆分为对角与非对角部分。这个拆分并非是 trivial 的，后面需要具体地讨论。

> **缺失格点偏置导数**
> 
> 上面的讨论在理想的全空间坐标积分下成立。现实里，在不少情况下也确实可以给出合理的 Hessian (指杂化 LDA/GGA 泛函)。
> 
> 但作为数值积分方法，我们没有考虑原子坐标变化对格点 $\bm{r}_g$ 与权重 $w_g$ 的影响。meta-GGA 的 Hessian 计算一般要求引入格点偏置的影响。我们将在其他文档中讨论该问题。

> **Column-major 维度约定**
>
> DFT 格点导数所涉及到的张量，一般来说，其维度依最连续 (column-major 下是最左侧的维度) 排序是
> - 格点维度 `g`，维度大小 `ngrids` $n_\mathrm{grids}$；
> - 基组维度 `u, v`，维度大小 `nao` $n_\mathrm{basis}$；
> - 参量维度 `x, y`，维度大小 `nvar` $n_\mathrm{var}$；
> - 空间维度 `t, s`，维度大小 3；存在特例为对称合并维度 `ts`，维度大小 6；
> - 基组格点导数维度 `*`，维度大小 `ncomp`；
> - 原子维度 `A, B`，维度大小 `natm` $n_\mathrm{atom}$。

## 2. 电子积分原子核偏导数的常用技巧

上面的讨论适用于任意性质。现在作特化：我们仅考虑 $\mathbb{A} = A_t$ 的情形；其中 $A_t$ 是指原子 $A$ 的 $t$ 三维空间分量。正常情况下用 $\bm{R}_A$ 向量 (或其三维分量 $R_{At}$) 表示，但为了程序实现对应上的便利，我们有时简化为记号 $A_t$。

一个常见技巧是，**原子轨道基**下，

$$
\partial_{A_t} \phi_\mu = - \partial_t \phi_\mu \delta_{\mu \in A} = - \phi_\mu^t \delta_{\mu \in A}
$$

大致说来，这利用了基函数只通过相对坐标 $\bm{r} - \bm{R}_A$ 依赖电子坐标与原子核坐标，由链式法则使得原子核坐标偏导可以转化为电子坐标偏导，但被偏导对象只能是特定原子上展开的基函数，不是所有基函数。

我们通常不称 DFT 格点积分为电子积分 (电子积分一般是指 1-2 电子、2-4 中心的简单算符解析积分)；但这不影响 DFT 作为 1 电子积分的本质。这里的技巧不仅在普通电子积分适用，在 DFT 也一样如此。

> **该技巧只针对原子轨道基有效**
>
> 该技巧只针对原子轨道基有效。这里的原子轨道不限于 Gaussian 基函数，也可以是 Slater 基函数、数值基函数等，以及其对应的赝势基。只要是原子轨道基，电子积分的原子核偏导数都可以转化为电子坐标偏导数。
>
> 但另一大类基函数是平面波。对于平面波，基函数导数一般应该化为动量空间下的算符操作，有其他的实现策略；且当前的技巧对平面波基函数完全不适用。

我们要注意到，在标度上，原子尽管数量很少，但也是一个标度。在二阶梯度 Skeleton 导数问题里，一种比较有效的策略是，先处理电子导数；不能进一步处理的部分，再处理原子核导数。不过这也因问题而异：像这里我们在 `fxc` 上，就先处理了原子导数；但在 `vxc` 上，我们先处理了电子导数。这在后面的实现细节里会有体现。

## 3. `fxc` 贡献项的实现细节

### 3.1 `fxc` 最终结算过程

我们先考察 `fxc` 的最终结算。首先，其贡献项一共是 4 项：

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s} \leftarrow \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} (\partial_{A_t} \xi_g^\chi) (\partial_{B_s} \xi_g^{\chi'})
\quad (\texttt{de\_fxc})
$$

**函数 `get_de_fxc`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `wf` | $w_g f_g^{\chi \chi'}$ | $(g, \chi, \chi')$</br>`[g, x, y]` | `[ngrids, nvar, nvar]` | $\mathrm{sym} (\chi, \chi')$ |
| `drho` | $\partial_{A_t} \xi_g^\chi$ | $(g, \chi, t, A)$</br>`[g, x, t, A]` | `[ngrids, nvar, 3, natm]` | |
| `de_fxc`</br>(output) | | $(t, s, A, B)$ </br>`[t, s, A, B]` | `[3, 3, natm, natm]` | $\mathrm{sym} (tA, sB)$ |

其程序实现，如果是在 NumPy 中，或引入 tblis 的 RSTSR，是可以比较方便地使用 einsum 给出的。

```python
# Please note the indices interchange of row-major / col-major
# This is numpy code, so row-major is used, different to col-major of above table
de_fxc = np.einsum("g, xyg, Atxg, Bsyg -> ABts", weights, fxc, drho, drho)
```

不使用 einsum 也是容易的，不过对张量维度的 broadcasting 小心一点即可。

我们要注意到，$\partial_{A_t} \xi_g^\chi$ 与 $\partial_{B_s} \xi_g^{\chi'}$ 实际上是一个东西，只是用了不同的角标而已。

该计算最大计算量是 $2 \times 3^2 n_\mathrm{atom}^2 n_\mathrm{var} n_\mathrm{grids}$ FLOPs，是较大内存瓶颈但计算量很小的 $O(N^3)$ 复杂度。

上式的 `wf` 是很容易获得的：$w_g$ 是基本的格点参数，$f_g^{\chi \chi'}$ 可以通过封装后的 LibXC 接口立即获得。困难的部分是 `drho` $\partial_{A_t} \xi_g^\chi$ 的计算。

### 3.2 密度格点一阶 Skeleton 梯度 `drho` 概述与实现决策

首先回顾密度格点定义：

$$
\xi_g^\chi = \sum_{\mu \nu} D_{\mu \nu} \xi_{g \mu \nu}^\chi
$$

对其作核坐标的 Skeleton 偏导数 $\partial_{A_t}$，将只会对 $\xi_{g \mu \nu}^\chi$ 作偏导数。$\xi_{g \mu \nu}^\chi$ 对于 LDA/GGA/mGGA 有不同的定义；但大致上，它都表现为
1. 密度矩阵 $D_{\mu \nu}$ 与一个右矢 $\phi_{g \nu}^*$ 先作矩阵乘法并缩并指标 $\nu$。计算复杂度是 $O(n_\mathrm{basis}^2 n_\mathrm{grid})$ 即 $O(N^3)$。
2. 随后再与一个左矢 $\phi_{g \mu}^*$ 作数乘并缩并指标 $\mu$；但如何缩并需要对 LDA (RHO) / GGA (SIGMA) / MGGA (TAU) 分别处理。该步骤内存遍历次数很多，耗时其实不少，也是程序实现的难点；但它其实是 $O(n_\mathrm{basis} n_\mathrm{grid})$ 即 $O(N^2)$ 计算复杂度。

上面的第 1 步是简单矩阵乘法，其程序实现是 1 行，因此不引入新的函数：

$$
\bar{\phi}_{g \mu}^* = \sum_{\nu} D_{\mu \nu} \phi_{g \nu}^*
\quad (\texttt{ao\_dm0})
$$

- `ao_dm0` $\bar{\phi}_{g \mu}^*$ 与 `ao` $\phi_{g \mu}^*$ 接近，是 $(g, \mu, *)$ `[ngrids, nao, ncomp]` 维度张量。
- 在实际 GGA/MGGA 二阶梯度计算中，`ao_dm0` $\bar{\phi}_{g \mu}^*$ 中的上标 $*$ 最大是一阶梯度 (非导数、$x$/$y$/$z$) 的 4 种情况。而 `ao` $\phi_{g \mu}^*$ 的上标 $*$ 则最大会到三阶梯度的 20 种情况。同时参考函数 `get_hess_ncomp_ao_dm0`。
- 对于 GGA/MGGA，该步的 FLOPs 是 $8 n_\mathrm{basis}^2 n_\mathrm{grids}$。

```rust
let ao_dm0 = index!(ao, ..ncomp_ao_dm0) % &dm0;
```

> **实现决策：使用 AO 基的密度矩阵与轨道缩并**
>
> 决策：使用下述变量 `ao_dm0` $\bar{\phi}_{g \mu}^*$ 作为重要中间量。
>
> 放弃的其他可能性：使用占据分子轨道基 $\bar{\phi}_{g i}^* = \sum_{\nu} C_{\nu i} \phi_{g \nu}^*$ 作预缩并。
>
> 这里并没有作详细分析与测评；但出于下述考量，放弃该可能性：
> - 主要原因：`vxc` 对角贡献项 `dao_vxc_diag` 同样需要 `ao_dm0` $\bar{\phi}_{g \mu}^*$。这一项没有办法利用占据轨道比基组数更少得以加速：其必须要扩张到原子轨道基组才能作后续计算 (对 $\delta_{\mu \in A}$ 的原子切片缩并、以及格点偏置贡献对 $\mathscr{T}_{\mu}^{(ts)}$ 的复用，都要求保留 $\mu$ 指标)。
> - 主要原因：内存用量可以接受。我们如果允许 $20 n_\mathrm{basis} n_\mathrm{grids}$ 的 `ao` $\phi_{g \mu}^*$ 存储，那么 $4 n_\mathrm{basis} n_\mathrm{grids}$ 的 `ao_dm0` $\bar{\phi}_{g \mu}^*$ 也应是可以接受的。
> - 次要原因：程序的编写会变得复杂。如果是能量或响应问题，基函数格点 `ao` 或许与占据轨道缩并会更有优势 (在 nimatmul 模块中，有一些 `_bra_trans` 函数专门用于此类问题)。但梯度问题中，经常会出现针对特定原子的部分基组缩并 $\delta_{\mu \in A}$，程序编写上会比较麻烦。
> - 可能的改进：如果使用一次占据轨道缩并、与一次占据轨道展开，在基组比较大的时候 (超过 6-31G)，FLOPs 是比较小的 $16 n_\mathrm{occ} n_\mathrm{basis} n_\mathrm{grids}$。这一改进的可能性来源于 `drho` 的指标结构：`drho` 的指标是 $(g, \chi, t, A)$ 而不含 $\mu$，被缩并的一端取占据轨道或原子轨道基皆可；这与 `dao_vxc_diag` 必须保留 $\mu$ 指标的情况不同。但考虑到这个计算量相对于 Fock Skeleton 一阶梯度来说并不大，且要明确引入占据轨道而非更广义的密度矩阵，因此没有采用这种微小但引入代码复杂程度的性能优化。

### 3.3 `drho` 计算

**函数 `get_drho`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `xc_type` | | | `LDA` / `GGA` / `MGGA` | |
| `ao` | $\phi_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>4/10/10 |
| `ao_dm0` | $\bar{\phi}_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>1/4/4 |
| `aoslices` | | | `natm` |
| `drho`</br>(output) | $\partial_{A_t} \xi_g^\chi$ | $(g, \chi, t, A)$ </br>`[g, x, t, A]` | `[ngrids, nvar, 3, natm]` | |

表中 `ao`/`ao_dm0` 行的 `ncomp` 栏 (依 LDA/GGA/MGGA 三种情形给出) 指该函数实际读取的最大导数分量数；实际分配的 `ao` 张量取所有函数的最大需求，由 `get_hess_ao_deriv` 决定 (LDA 为至二阶导数共 10 个分量，GGA/MGGA 为至三阶共 20 个分量)。下文各函数表中的 `ncomp` 栏均按此约定。

**`drho`: LDA (RHO)**

$$
\begin{aligned}
\partial_{A_t} \xi_g^{\chi = \rho}
&= \sum_{\mu \nu} \partial_{A_t} \big( \phi_{g \mu} \phi_{g \nu} D_{\mu \nu} \big) \quad (\texttt{drho[:, 0, :, :]}) \\
&= - 2 \sum_{\mu} \delta_{\mu \in A} \phi_{g \mu}^t \bar{\phi}_{g \mu}
\end{aligned}
$$

上式的 2 倍，来源于偏导链式法则对 $\mu, \nu$ 的对称项的合并。

**`drho`: GGA (SIGMA)**

$$
\begin{aligned}
\partial_{A_t} \xi_g^{\chi = \rho_r}
&= 2 \sum_{\mu \nu} \partial_{A_t} \big( \phi_{g \mu}^r \phi_{g \nu} D_{\mu \nu} \big) \quad (\texttt{drho[:, 1:4, :, :]}) \\
&= - 2 \sum_{\mu} \delta_{\mu \in A} \left( \phi_{g \mu}^{t r} \bar{\phi}_{g \mu} + \phi_{g \mu}^t \bar{\phi}_{g \mu}^r \right)
\end{aligned}
$$

请留意上式的推导是跳步的。我们需要利用一些 $\mu \leftrightarrow \nu$ dummy 指标轮换简化到上式 (留作思考：为何上式的第二项不是 $\phi_{g \mu}^r \bar{\phi}_{g \mu}^t$？)。2 倍的系数来源于 $\phi_{g \mu}^r \phi_{g \nu} + \phi_{g \mu} \phi_{g \nu}^r$ 的对称性，这与 LDA (RHO) 的情况稍有不同。

**`drho`: MGGA (TAU)**

$$
\begin{aligned}
\partial_{A_t} \xi_g^{\chi = \rho_\tau}
&= \frac{1}{2} \sum_{r \mu \nu} \partial_{A_t} \big( \phi_{g \mu}^r \phi_{g \nu}^r D_{\mu \nu} \big) \quad (\texttt{drho[:, 4, :, :]}) \\
&= - \sum_{r \mu} \delta_{\mu \in A} \phi_{g \mu}^{t r} \bar{\phi}_{g \mu}^r
\end{aligned}
$$

这里最后的表达式就没有 2 倍系数了；这单纯是因为 $\tau$ 的定义里有 $\frac{1}{2}$ 的系数。

> **实现决策：`drho` 实现中 $\delta_{\mu \in A}$ 的应用时机**
>
>上述计算有两种实现策略：
>
> 1. 直接依公式实现，原地对指标 $\mu \in A$ 执行缩并；
> 2. 先生成一个含有指标 $\mu$ 的、维度为 $(g, \mu, \chi, t)$ 的张量，基于此应用 $\delta_{\mu \in A}$ 执行缩并，得到最终维度 $(g, \chi, t, A)$ 的 `drho` 张量。
>
> 我们最终选择第 1 种实现。原因是，第 1 种实现的写入 memory footprint 较小，不会有数倍的 $(\mu, g)$ 的大内存写入。第 2 种尽管是 RI-JK 或后面 vxc 贡献项计算所用到的策略，但在这里反而不适合。

## 4. `vxc` 对角部分实现细节

`vxc` 贡献项的核心是如何构建与处理下述缩并问题：

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s} \leftarrow \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial A_t \partial B_s}
$$

### 4.1 `vxc` 对角与非对角部分的定义

仍然先回顾密度格点的定义：

$$
\xi_g^\chi = \sum_{\mu \nu} D_{\mu \nu} \xi_{g \mu \nu}^\chi
$$

我们将要对其中的 $\xi_{g \mu \nu}^\chi$ 作两次偏导。根据 $\mu, \nu$ 的不同组合，我们将 `vxc` 贡献项拆分为两部分：

- **对角部分 (`diag`)**：指对其中一个原子轨道 $\mu$ 求两次导数；由原子核导数的规则，这个导数必须是对同一个原子求两次导数。利用对称性，$\nu$ 的导数贡献与 $\mu$ 一致。
- **非对角部分 (`nondiag`)**：则是对两个原子轨道 $\mu, \nu$ 各求一次导数；这两个导数可以在不同的原子上进行。

需要指出，对角部分确实只会对维度为 $(t, s, A, B)$ Hessian 张量的 $A = B$ 部分作贡献，但 **非对角部分也会对 $A = B$ 的部分作贡献**，只是贡献的具体项不同。这里用“对角”与“非对角”对意义不同的部分作区分，或许是不幸，但暂时没有更好的命名。

DFT 相比于 J/K 积分，
- DFT 是 2 中心而 J/K 是 4 中心，要考虑的原子轨道导数组合少了很多；
- DFT 要区分 LDA/GGA/MGGA、且表达式非线性，要考虑的项又多了不少。

### 4.2 对角部分中间量 `dao_vxc_diag` $\mathscr{T}_{\mu}^{(ts)}$ 的引入

这里我们采用先处理电子导数，再处理原子核导数的策略。留意由于我们两次偏导了原子核，两个负号相互抵消了。

与刚才求一阶密度的方式不同，我们不提前缩并所有原子轨道指标，而是将与原子有关的 $\delta_{\mu \in A}$ 放在最后进行。回顾到

$$
\xi_g^\chi = \sum_{\mu \nu} D_{\mu \nu} \xi_{g \mu \nu}^\chi
$$

留意我们是要对单个原子作偏导；考虑到 $(\mu, \nu)$ 的对称性，下式有两倍，但偏导只作用在 $\mu$ 上。由于公式表达比较困难，我们只能在这里用文字描述限制偏导的行为。

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial A_s} \leftarrow 2 \sum_{g \chi \mu \nu} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} D_{\mu \nu} \delta_{\mu \in A} \quad \text{(restrict $\partial$ to $\mu$, \texttt{de\_vxc\_diag})}
$$

我们会注意到，上式中的 $\nu$ 并不参与偏导，因此提前对 $\nu$ 作边际化 (就是前面 `fxc` 计算时的中间量 `ao_dm0` $\bar{\phi}_{g \mu}^{*}$)，可以将原先 `gu, gv -> uv` 的问题化为 `gu, gu -> u` 的问题，节省一次较大的 $O(n_\mathrm{basis}^2 n_\mathrm{grids})$ 矩阵乘法计算。

同时，该导数关于 $t, s$ 是对称的。因此，原先 $3 \times 3$ 的问题可以化为 $(xx, xy, xz, yy, yz, zz)$ 的 6 分量问题，稍微节省一些计算量。

因此，我们将会给出一个中间量 `dao_vxc_diag` $\mathscr{T}_{\mu}^{(ts)}$ (维度 $(\mu, (ts))$，大小 $(n_\mathrm{basis}, 6)$)：

$$
\mathscr{T}_{\mu}^{(ts)} = 2 \sum_{g \chi \nu} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} D_{\mu \nu} \quad \text{(restrict $\partial$ to $\mu$, \texttt{dao\_vxc\_diag})}
$$

随后引入原子依赖的求和计算：

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial A_s} \leftarrow \sum_\mu \mathscr{T}_{\mu}^{(ts)} \delta_{\mu \in A} \quad (\texttt{de\_vxc\_diag})
$$

很显然上面一步是没有什么计算量的。至此，我们将问题化归为如何求取 `dao_vxc_diag` $\mathscr{T}_{\mu}^{(ts)}$ 中间量。在计算复杂度分析上，它与 `fxc` 倒是非常相似：其最大的计算量在 `ao_dm0` $\bar{\phi}_{g \mu}$ 的计算上，剩下的是大量复杂的 $O(N^2)$ 缩并。

在继续讨论前，我们指出，$w_g f_g^\chi$ 总是成对出现的。因此，我们可以先将 $w_g$ 与 $f_g^\chi$ 相乘，存储到 `wv` 或 `wvxc` 变量中。

**函数 `make_dao_vxc_diag`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `xc_type` | | | `LDA` / `GGA` / `MGGA` | |
| `ao` | $\phi_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>10/20/20 |
| `ao_dm0` | $\bar{\phi}_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>1/4/4 |
| `wv` | $w_g f_g^\chi$ | $(g, \chi)$</br>`[g, x]` | `[ngrids, nvar]` | |
| `dao_vxc_diag`</br>(output) | $\mathscr{T}_{\mu}^{(ts)}$ | $(\mu, (ts))$</br>`[u, ts]` | `[nao, 6]` | |

**函数 `get_de_vxc_diag`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `dao_vxc_diag` | $\mathscr{T}_{\mu}^{(ts)}$ | $(\mu, (ts))$</br>`[u, ts]` | `[nao, 6]` | |
| `aoslices` | | | `natm` |
| `de_vxc_diag`</br>(output) | | $(t, s, A, B)$</br>`[t, s, A, B]` | `[3, 3, natm, natm]` | $A \neq B$ 零值 |

> **实现决策：分离 `dao_vxc_diag` 与 `de_vxc_diag` 为不同的函数实现**
>
> 如果仅仅是 Skeleton 能量二阶导数，实际上 `dao_vxc_diag` 与 `de_vxc_diag` 可以合并为一个函数实现。但考虑到下面两个因素，这两个函数将分开实现：
> - 分开实现并没有引入额外的计算量。
> - 涉及到格点偏置的计算中，函数 `contract_pvxc` 需要 `dao_vxc_diag` 的中间量 $\mathscr{T}_{\mu}^{(ts)}$。这点对 `dao_vxc_off` 的中间量 $\mathscr{T}_{\mu \nu}^{ts}$ 也同样适用。为了代码复用，`dao_vxc_diag` 与 `de_vxc_diag` 分开实现是比较合理的。

### 4.3 `dao_vxc_diag` 的具体公式与实现

下面我们对 LDA (RHO), GGA (SIGMA), MGGA (TAU) 分别给出 `dao_vxc_diag` 的具体公式。我们注意，对于 GGA 与 MGGA 任务，下述贡献项都需要叠加在一起，即 GGA 需要 RHO + SIGMA，MGGA 需要 RHO + SIGMA + TAU。

下述计算中，除了作为输入参数的 `ao_dm0` 本身计算复杂度是 $O(n_\mathrm{basis}^2 n_\mathrm{grid})$ 即 $O(N^3)$ (但由于是输入参数，不计入当前函数的实际计算量)，其余的计算量都是较大量的 $O(n_\mathrm{basis} n_\mathrm{grid})$ 即 $O(N^2)$。

**`dao_vxc_diag`: LDA (RHO)**

LDA 部分对应 $\chi = \rho$，$\xi_{g \mu \nu}^{\chi = \rho} = \phi_{g \mu} \phi_{g \nu}$。仅对 $\mu$ 作偏导，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \rho}}{\partial t \partial s} = \phi_{g \mu}^{ts} \phi_{g \nu} \quad \text{(restrict $\partial$ to $\mu$)}
$$

代入 $\mathscr{T}_\mu^{(ts)}$ 的定义，并将 $\nu$ 缩并到 `ao_dm0` $\bar{\phi}_{g \mu} = \sum_\nu \phi_{g \nu} D_{\mu \nu}$ 上：

$$
\mathscr{T}_{\mu}^{(ts)} \mathrel{+}= 2 \sum_g w_g f_g^{\rho} \phi_{g \mu}^{ts} \bar{\phi}_{g \mu}
$$

程序上，$\phi_{g \mu}^{ts}$ 是 `ao` 张量在 $ts \in \{xx, xy, xz, yy, yz, zz\}$ 上的 6 个分量；$w_g f_g^\rho$ 即 `wv[0]`。

**`dao_vxc_diag`: GGA (SIGMA)**

GGA 部分对应 $\chi = \rho^r$，$\xi_{g \mu \nu}^{\chi = \rho^r} = \phi_{g \mu}^r \phi_{g \nu} + \phi_{g \mu} \phi_{g \nu}^r$。仅对 $\mu$ 作偏导，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \rho^r}}{\partial t \partial s} = \phi_{g \mu}^{tsr} \phi_{g \nu} + \phi_{g \mu}^{ts} \phi_{g \nu}^r \quad \text{(restrict $\partial$ to $\mu$)}
$$

代入 $\mathscr{T}_\mu^{(ts)}$ 的定义。第一项以 $\bar{\phi}_{g \mu} = \sum_\nu \phi_{g \nu} D_{\mu \nu}$ 缩并 $\nu$，第二项以 $\bar{\phi}_{g \mu}^r = \sum_\nu \phi_{g \nu}^r D_{\mu \nu}$ 缩并 $\nu$：

$$
\mathscr{T}_{\mu}^{(ts)} \mathrel{+}= 2 \sum_{g r} w_g f_g^{\rho^r} \left( \phi_{g \mu}^{tsr} \bar{\phi}_{g \mu} + \phi_{g \mu}^{ts} \bar{\phi}_{g \mu}^r \right)
$$

其中：

- 第一项需要 $\phi_{g \mu}^{tsr}$ 即原子轨道的三阶电子坐标导数。具体而言，对每个 $(ts)$ 分量需要取 `ao` 中的三阶导分量 (例如 $(ts) = (xx)$ 时对应 $\{xxx, xxy, xxz\}$)；其与 `wv[1:4]` $w_g f_g^{\rho^r}$ ($r \in \{x, y, z\}$) 相乘后再与 $\bar{\phi}_{g \mu}$ 缩并。
- 第二项中，$\bar{\phi}_{g \mu}^r$ 即 `ao_dm0` 的 $r \in \{x, y, z\}$ 分量；其与 `wv[1:4]` 相乘后求和，再与 $\phi_{g \mu}^{ts}$ 缩并。事实上，该缩并可以与 LDA (RHO) 部分共用 $\phi_{g \mu}^{ts}$ 的访问，从而合并为一次缩并，节省计算量。

**`dao_vxc_diag`: MGGA (TAU)**

MGGA (TAU) 部分对应 $\chi = \tau$，$\xi_{g \mu \nu}^{\chi = \tau} = \frac{1}{2} \sum_r \phi_{g \mu}^r \phi_{g \nu}^r$。仅对 $\mu$ 作偏导，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \tau}}{\partial t \partial s} = \frac{1}{2} \sum_r \phi_{g \mu}^{tsr} \phi_{g \nu}^r \quad \text{(restrict $\partial$ to $\mu$)}
$$

代入 $\mathscr{T}_\mu^{(ts)}$ 的定义，以 $\bar{\phi}_{g \mu}^r = \sum_\nu \phi_{g \nu}^r D_{\mu \nu}$ 缩并 $\nu$，并与 $\mathscr{T}_\mu^{(ts)}$ 定义中的 $2$ 系数相消：

$$
\mathscr{T}_{\mu}^{(ts)} \mathrel{+}= \sum_{g r} w_g f_g^\tau \phi_{g \mu}^{tsr} \bar{\phi}_{g \mu}^r
$$

程序上的对应与 GGA 第一项类似：对每个 $(ts)$ 分量，从 `ao` 中取出 $\phi_{g \mu}^{tsr}$ 共 3 个三阶导分量，与 `ao_dm0` 的 $r \in \{x, y, z\}$ 分量逐项配对，并以 `wv[4]` $w_g f_g^\tau$ 加权后缩并。同样是 $O(n_\mathrm{basis} n_\mathrm{grid})$ 复杂度。

## 5. `vxc` 非对角部分实现细节

### 5.1 非对角部分中间量 `dao_vxc_off` $\mathscr{T}_{\mu \nu}^{ts}$ 的引入

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s} \leftarrow \sum_{g \chi \mu \nu} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} D_{\mu \nu} \delta_{\mu \in A} \delta_{\nu \in B} + \text{swap} (A_t, B_s) \quad \text{(restrict $\partial_t$ to $\mu$, $\partial_s$ to $\nu$)}
$$

非对角部分需要对两个原子轨道 $\mu, \nu$ 各求一次导数。与对角情况不同，这时我们不再适合先边际掉其中一个原子轨道，而是需要保留所有原子轨道。我们将定义下述中间量 `dao_vxc_off` $\mathscr{T}_{\mu \nu}^{t s}$ (维度 $(\mu, \nu, t, s)$，大小 $(n_\mathrm{basis}, n_\mathrm{basis}, 3, 3)$；row-major 的 Python 参考实现中为 $(t, s, \mu, \nu)$ 即 `[3, 3, nao, nao]`)：

$$
\mathscr{T}_{\mu \nu}^{t s} = \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} + \text{swap} (t \mu, s \nu) \quad \text{(restrict $\partial_t$ to $\mu$, $\partial_s$ to $\nu$, \texttt{dao\_vxc\_off})}
$$

那么最终的二阶导数贡献项可以写为：

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s} = \sum_{\mu \nu} \mathscr{T}_{\mu \nu}^{t s} D_{\mu \nu} \delta_{\mu \in A} \delta_{\nu \in B} \quad (\texttt{de\_vxc\_off})
$$

请留意，$\mathscr{T}_{\mu \nu}^{t s}$ 的定义中已经完成了 $\text{swap} (t \mu, s \nu)$ 的对称化。因此，上式的单次缩并 (配合 $\partial_t$ 限制在 $\mu$、$\partial_s$ 限制在 $\nu$ 的约定) 已经给出完整的非对角贡献，不需要再额外补 $\text{swap} (A_t, B_s)$ 项。程序中，$(B, A)$ 块直接由 $(A, B)$ 块转置填充；这只是存储上的镜像操作，并非额外的求和项。

**函数 `make_dao_vxc_off`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `xc_type` | | | `LDA` / `GGA` / `MGGA` | |
| `ao` | $\phi_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>4/10/10 |
| `wv` | $w_g f_g^\chi$ | $(g, \chi)$</br>`[g, x]` | `[ngrids, nvar]` | |
| `dao_vxc_off`</br>(output) | $\mathscr{T}_{\mu \nu}^{t s}$ | $(\mu, \nu, t, s)$</br>`[u, v, 3, 3]` | `[nao, nao, 3, 3]` | |

**函数 `get_de_vxc_off`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `dao_vxc_off` | $\mathscr{T}_{\mu \nu}^{ts}$ | $(\mu, \nu, t, s)$</br>`[u, v, 3, 3]` | `[nao, nao, 3, 3]` | |
| `dm0` | $D_{\mu \nu}$ | $(\mu, \nu)$</br>`[u, v]` | `[nao, nao]` | |
| `aoslices` | | | `natm` |
| `de_vxc_off`</br>(output) | | $(t, s, A, B)$</br>`[t, s, A, B]` | `[3, 3, natm, natm]` | |

### 5.2 `dao_vxc_off` 的具体公式与实现

**`dao_vxc_off`: LDA (RHO)**

LDA 部分对应 $\chi = \rho$，$\xi_{g \mu \nu}^{\chi = \rho} = \phi_{g \mu} \phi_{g \nu}$。$\partial_t$ 仅作用在 $\mu$、$\partial_s$ 仅作用在 $\nu$，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \rho}}{\partial t \partial s} = \phi_{g \mu}^t \phi_{g \nu}^s \quad \text{(restrict $\partial_t$ to $\mu$, $\partial_s$ to $\nu$)}
$$

代入 $\mathscr{T}_{\mu \nu}^{ts}$ 的定义：

$$
\mathscr{T}_{\mu \nu}^{ts} \mathrel{+}= \sum_g w_g f_g^{\rho} \phi_{g \mu}^t \phi_{g \nu}^s + \text{swap} (t \mu, s \nu)
$$

程序上，这是一组 9 个 $(n_\mathrm{basis}, n_\mathrm{grids})$ 将格点指标边际掉的矩阵乘法。其中 $\phi^t$ 即 `ao` 张量在 $t \in \{x, y, z\}$ 上的 3 个分量；$w_g f_g^\rho$ 即 `wv[0]`。

我们注意到 $\mathscr{T}_{\mu \nu}^{ts}$ 关于 $(t \leftrightarrow s, \mu \leftrightarrow \nu)$ 的联合交换是对称的 (因为 $\xi_{g \mu \nu}^\chi = \xi_{g \nu \mu}^\chi$，而 $(t, s)$ 仅作为对称记号)。因此原先 $3 \times 3 = 9$ 的矩阵乘法原则上可以只算 $6$ 个 $(ts) \in \{xx, xy, xz, yy, yz, zz\}$ 分量，剩下的 $(yx, zx, zy)$ 通过 AO 指标转置补齐。不过，当前实现中 LDA 部分仍按 $9$ 个分量完整计算，以与 GGA 部分的循环结构保持一致。

该项的最大计算量是 $3^2 \times 2 n_\mathrm{basis}^2 n_\mathrm{grid}$ FLOPs，是 $O(N^3)$ 复杂度。

**`dao_vxc_off`: GGA (SIGMA)**

GGA 部分对应 $\chi = \rho^r$，$\xi_{g \mu \nu}^{\chi = \rho^r} = \phi_{g \mu}^r \phi_{g \nu} + \phi_{g \mu} \phi_{g \nu}^r$。$\partial_t$ 仅作用在 $\mu$、$\partial_s$ 仅作用在 $\nu$，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \rho^r}}{\partial t \partial s} = \phi_{g \mu}^{t r} \phi_{g \nu}^s + \phi_{g \mu}^t \phi_{g \nu}^{s r} \quad \text{(restrict $\partial_t$ to $\mu$, $\partial_s$ to $\nu$)}
$$

代入 $\mathscr{T}_{\mu \nu}^{ts}$ 的定义：

$$
\mathscr{T}_{\mu \nu}^{ts} \mathrel{+}= \sum_{g r} w_g f_g^{\rho^r} \left( \phi_{g \mu}^{t r} \phi_{g \nu}^s + \phi_{g \mu}^t \phi_{g \nu}^{s r} \right) + \text{swap} (t \mu, s \nu)
$$

注意到 $\mathscr{T}_{\mu \nu}^{ts}$ 在联合交换 $(t \leftrightarrow s, \mu \leftrightarrow \nu)$ 下对称，而上式第二项即为第一项在该联合交换下的像。因此**只需算第一项**，最后做一次转置-对称化即可得到第二项的贡献。

进一步注意到，第一项与 LDA (RHO) 的右因子 $\phi_{g \nu}^s$ 是相同的；只是左因子从 $\phi_{g \mu}^t$ 变为 $\phi_{g \mu}^{tr}$ 并对 $r$ 求和。因此，我们可以将 LDA 的左因子与 GGA 的左因子先合并为一个**带权 ket 端中间量**，再统一与 $\phi_{g \nu}^s$ 做矩阵乘法：

$$
\widetilde{\phi}_{g \mu}^{t} = \frac{1}{2} w_g f_g^\rho \phi_{g \mu}^t + \sum_r w_g f_g^{\rho^r} \phi_{g \mu}^{t r} \quad (\texttt{aowv})
$$

注意第一项中 $\frac{1}{2}$ 的来源：LDA 项 $2 w_g f_g^\rho \phi_{g \mu}^t \phi_{g \nu}^s$ 经过下面对称化会被算两次，因此需要预除以 $2$ 才能与 GGA 项合在一起做一次对称化。GGA 项 $2 w_g f_g^{\rho^r} \phi_{g \mu}^{tr} \phi_{g \nu}^s$ 自身只贡献一次 (第二项由对称化补出)，因此 $\widetilde{\phi}$ 内不需要 $\frac{1}{2}$ 系数。

随后做矩阵乘法 (留意角标 $t, s$ 与 $\mu, \nu$ 的对应顺序)，并作对称化：

$$
\mathscr{T}_{\mu \nu}^{ts} = 2 \sum_g \widetilde{\phi}_{g \mu}^t \phi_{g \nu}^s + \text{swap} (t \mu, s \nu)
$$

至此 LDA + GGA 部分的 $\mathscr{T}_{\mu \nu}^{ts}$ 全部完成。注意这里我们将 LDA (RHO) 与 GGA (SIGMA) 合并在一起算，是为了节省一组 ket 端 $\phi_{g \nu}^s$ 的矩阵乘法；这与第 4 节 `dao_vxc_diag` 中将 LDA 项与 GGA 第二项合并为一次缩并的优化是同样的策略。

该项的最大计算量出现在 9 个 $(\widetilde{\phi}^t)^\dagger \phi^s$ 的矩阵乘法上，即 $3^2 \times 2 n_\mathrm{basis}^2 n_\mathrm{grid}$ FLOPs，是 $O(N^3)$ 复杂度。这里 (与 LDA 的实际实现一样) 没有利用 $(ts) \leftrightarrow (st)$ 的对称性，是因为 $\widetilde{\phi}^t$ 内部混合了 LDA 与 GGA 的项，难以直接利用 $(ts)$ 对称性；而对称化反而是放在矩阵乘法之后通过转置完成的。

**`dao_vxc_off`: MGGA (TAU)**

MGGA (TAU) 部分对应 $\chi = \tau$，$\xi_{g \mu \nu}^{\chi = \tau} = \frac{1}{2} \sum_r \phi_{g \mu}^r \phi_{g \nu}^r$。$\partial_t$ 仅作用在 $\mu$、$\partial_s$ 仅作用在 $\nu$，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \tau}}{\partial t \partial s} = \frac{1}{2} \sum_r \phi_{g \mu}^{t r} \phi_{g \nu}^{s r} \quad \text{(restrict $\partial_t$ to $\mu$, $\partial_s$ to $\nu$)}
$$

代入 $\mathscr{T}_{\mu \nu}^{ts}$ 的定义：

$$
\mathscr{T}_{\mu \nu}^{ts} \mathrel{+}= \frac{1}{2} \sum_{g r} w_g f_g^\tau \phi_{g \mu}^{t r} \phi_{g \nu}^{s r} + \text{swap} (t \mu, s \nu)
$$

上式与 LDA 一样具有 $(t \leftrightarrow s, \mu \leftrightarrow \nu)$ 的联合交换对称性 (注意单个 $(t, s)$ 分块本身并不对称，而是 $\mathscr{T}^{ts} = (\mathscr{T}^{st})^{\mathsf{T}}$，转置发生在 AO 指标上)，因此仍然可以使用 $(ts) \in \{xx, xy, xz, yy, yz, zz\}$ 的 6 分量来计算，剩下的 $(yx, zx, zy)$ 通过 AO 指标转置补齐。该计算量会比较大，涉及到 $6 \times 3 \times 2 n_\mathrm{basis}^2 n_\mathrm{grid}$ FLOPs，是 $O(N^3)$ 复杂度；其中，$6$ 来源于 $(ts)$ 的分量数，$3$ 来源于 $r$ 的分量数。

> **实现决策：避免二阶导数格点的直接计算**
>
> `vxc` 非对角部分应该是计算量最大的部分了。我们这里采用的是 $(\mu, \nu, t, s)$ 中间量策略，在 mGGA 下需要 27 个 $2 n_\mathrm{basis}^2 n_\mathrm{grid}$ FLOPs 的矩阵乘法。这个计算量要大于前面的 `fxc` 与 `vxc` 对角部分不少。
>
> 这个问题是否有其他解决方法？最直观的方法是，采用与 `fxc` 一样的策略，直接计算得到二阶导数密度格点 $\partial_{A_t} \partial_{B_s} \xi^\chi_g$ 即 $(n_\mathrm{atm}, n_\mathrm{atm}, 3, 3, n_\mathrm{var}, n_\mathrm{grids})$ 的张量，随后与 vxc 作缩并。具体实现上当然可以分批给出 $\partial_{A_t} \partial_{B_s} \xi^\chi_g$ 的分量，但其包含了格点数量，且导数密度格点会是被缩并维度非常小 (一个原子的原子轨道数量) 的长条形矩阵乘法，对缓存可能不算太友好。同时，FMA 的数量我相信与上述中间量策略是差不多的。