# DFT 二阶 Skeleton 导数

这份文档将讨论 DFT 二阶 Skeleton 导数的实现策略。我们只使用矩阵乘法的策略，不使用其他优化工具。

## 1. 二阶 Skeleton 导数：概论

我们将 DFT 二阶 Skeleton 导数分为三部分：
- `fxc` 即 $f^{\chi \chi'}$ 相关部分；
- `vxc` 对角部分 (单原子双重导数)；
- `vxc` 普通部分 (双原子各一重导数)。

我们这里稍作展开。首先，Skeleton 导数的定义是，在密度矩阵 (轨道系数) 不发生变化的情况下，改变原子坐标的导数。Skeleton 导数可以通过固定轨道系数、改变原子坐标，作数值差分计算得到。

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
\end{aligned}
$$

到这里，我们已经可以将第一项 (`fxc` 贡献项) 拆分出来了。第二项是 `vxc` 贡献，取决于具体的偏导计算过程，我们将其拆分为对角与普通部分。这个拆分并非是 trivial 的，后面需要具体地讨论。

## 2. 原子核偏导数的常用技巧

上面的讨论适用于任意性质。现在作特化：我们仅考虑 $\mathbb{A} = A_t$ 的情形；其中 $A_t$ 是指原子 $A$ 的 $t$ 三维空间分量。正常情况下用 $\bm{R}_A$ 向量表示，但为了程序实现对应上的便利，我们就用简化记号 $A_t$。

一个常见技巧是，**原子轨道基**下 (其他基有其他处理技巧)，

$$
\partial_{A_t} \phi_\mu = - \partial_t \phi_\mu \delta_{\mu \in A} = - \phi_\mu^t \delta_{\mu \in A}
$$

大致说来，这使用到了空间上的 Stokes 定理，使得原子核坐标偏导可以转化为电子坐标偏导，但被偏导对象只能是特定基函数。

我们要注意到，在标度上，原子尽管数量很少，但也是一个标度。在二阶梯度 Skeleton 导数问题里，一种比较有效的策略是，先处理电子导数；不能进一步处理的部分，再处理原子核导数。不过这也因问题而异：像这里我们在 `fxc` 上，就先处理了原子导数；但在 `vxc` 上，我们先处理了电子导数。这在后面的实现细节里会有体现。

## 3. `fxc` 贡献项的实现细节

### 3.1 `fxc` 最终结算过程

我们先考察 `fxc` 的最终结算。首先，其贡献项一共是 4 项：

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s} \leftarrow \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} (\partial_{A_t} \xi_g^\chi) (\partial_{B_s} \xi_g^{\chi'})
$$

上式用 einsum 可以直接写出。我们在 PySCF 语境 (row-major) 下，先回顾与总结上式的记号与维度：

| 表达式 | 程序变量 | 维度 | einsum 记号 |
|--|--|--|--|
| $w_g$ | `weights` | $(g)$ <br> `[ngrids]` | `g` |
| $f_g^{\chi \chi'}$ | `fxc` | $(\chi, \chi', g)$ <br> `[nvar, nvar, ngrids]` | `gxy` |
| $\partial_{A_t} \xi_g^\chi$ | `drho` | $(A, t, \chi, g)$ <br> `[natm, 3, nvar, ngrids]` | `Atxg` |
| $\partial_{B_s} \xi_g^{\chi'}$ | `drho` | $(B, s, \chi', g)$ <br> `[natm, 3, nvar, ngrids]` | `Bsyg` |

那么上式就可以用非常简单的计算过程给出：

```python
de_fxc = np.einsum("g, xyg, Atxg, Bsyg -> ABts", weights, fxc, drho, drho)
```

不使用 einsum 也是容易的，不过要对张量维度的 broadcasting 要小心一点即可。

我们要注意到，$\partial_{A_t} \xi_g^\chi$ 与 $\partial_{B_s} \xi_g^{\chi'}$ 实际上是一个东西，只是用了不同的角标而已。

该计算最大计算量是 $2 \times 3^2 n_\mathrm{atom}^2 n_\mathrm{var} n_\mathrm{grids}$ FMAs，是较大内存瓶颈的 $O(N^3)$ 复杂度。

上面唯一困难的部分是 `drho` $\partial_{A_t} \xi_g^\chi$ 的计算。

- 其最大计算复杂度是 $O(n_\mathrm{AO}^2 n_\mathrm{grid})$ 即 $O(N^3)$，出现在 `ao_dm0` $\bar{\phi}_{g \mu}$ 的计算上 (定义在下面给出)。
- 缩并 $\delta_{\mu \in A}$ 到 $\partial_{A_t} \xi_g^\chi$ 内存遍历次数很多，耗时其实不少，也是程序实现的难点；但它其实是 $O(n_\mathrm{AO} n_\mathrm{grid})$ 即 $O(N^2)$ 计算复杂度。原子尽管也参与了计算，但注意到 $\delta_{\mu \in A}$，因此在计算复杂度核算时，原子数可以合并到原子轨道里。

这里我们需要对 LDA (RHO), GGA (SIGMA), MGGA (TAU) 分别讨论。

### 3.2 LDA (RHO)

$$
\begin{aligned}
\partial_{A_t} \xi_g^{\chi = \rho}
&= \sum_{\mu \nu} \partial_{A_t} \big( \phi_{g \mu} \phi_{g \nu} D_{\mu \nu} \big) \quad (\texttt{drho[:, :, 0]}) \\
&= - 2 \sum_{\mu} \delta_{\mu \in A} \phi_{g \mu}^t \bar{\phi}_{g \mu}
\end{aligned}
$$

其中，我们定义

$$
\bar{\phi}_{g \mu} = \sum_\nu \phi_{g \nu} D_{\mu \nu} \quad (\texttt{ao\_dm0})
$$

上式的 2 倍，来源于偏导链式法则对 $\mu, \nu$ 的对称项的合并。

### 3.3 GGA (SIGMA)

$$
\begin{aligned}
\partial_{A_t} \xi_g^{\chi = \rho_r}
&= 2 \sum_{\mu \nu} \partial_{A_t} \big( \phi_{g \mu}^r \phi_{g \nu} D_{\mu \nu} \big) \quad (\texttt{drho[:, :, 1:4]}) \\
&= - 2 \sum_{\mu} \delta_{\mu \in A} \left( \phi_{g \mu}^{t r} \bar{\phi}_{g \mu} + \phi_{g \mu}^t \bar{\phi}_{g \mu}^r \right)
\end{aligned}
$$

请留意上式的推导是跳步的。我们需要利用一些 $\mu \leftrightarrow \nu$ dummy 指标轮换简化到上式 (留作思考：为何上式的第二项不是 $\phi_{g \mu}^r \bar{\phi}_{g \mu}^t$？)。2 倍的系数来源于 $\phi_{g \mu}^r \phi_{g \nu} + \phi_{g \mu} \phi_{g \nu}^r$ 的对称性，这与 LDA (RHO) 的情况稍有不同。

### 3.4 MGGA (TAU)

$$
\begin{aligned}
\partial_{A_t} \xi_g^{\chi = \rho_\tau}
&= \frac{1}{2} \sum_{r \mu \nu} \partial_{A_t} \big( \phi_{g \mu}^r \phi_{g \nu}^r D_{\mu \nu} \big) \quad (\texttt{drho[:, :, 4]}) \\
&= - \sum_{r \mu} \delta_{\mu \in A} \phi_{g \mu}^{t r} \bar{\phi}_{g \mu}^r
\end{aligned}
$$

这里最后的表达式就没有 2 倍系数了；这单纯是因为 $\tau$ 的定义里有 $\frac{1}{2}$ 的系数。

### 3.5 代码实现决定

上述计算有两种实现策略：

1. 直接依公式实现，原地对指标 $\mu \in A$ 执行缩并；
2. 先生成一个 $(t, \xi, \mu, g)$ 的张量，基于此对 $\mu \in A$ 执行缩并，得到 $(A, t, \xi, g)$ 的 `drho` 张量。

我们最终选择第 1 种实现。原因是，第 1 种实现的写入 memory footprint 较小，不会有数倍的 $(\mu, g)$ 的大内存写入。第 2 种尽管是 RI-JK 或后面 vxc 贡献项计算所用到的策略，但在这里反而不适合。

## 4. `vxc` 对角部分实现细节

`vxc` 贡献项的核心是如何构建与处理下述缩并问题：

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s} \leftarrow \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial A_t \partial B_s}
$$

### 4.1 `vxc` 对角与普通部分的定义

与 RI-JK 一样，DFT 也会涉及到对角与普通部分的差别。
- 对角部分是指对其中一个原子轨道 $\mu$ 求两次导数；由原子核导数的规则，这个导数必须是对同一个原子求两次导数。
- 普通部分则是对两个原子轨道 $\mu, \nu$ 各求一次导数；这两个导数可以在不同的原子上进行。

DFT 相比于 J/K 积分，
- DFT 是 2 中心而 J/K 是 4 中心，要考虑的原子轨道导数组合少了很多；
- DFT 要区分 LDA/GGA/MGGA，要考虑的项又多了不少。

### 4.2 对角部分中间量 `dao_vxc_diag`

这里我们采用先处理电子导数，再处理原子核导数的策略。留意由于我们两次偏导了原子核，两个负号相互抵消了。

与刚才求一阶密度的方式不同，我们不提前缩并所有原子轨道指标，而是将与原子有关的 $\delta_{\mu \in A}$ 放在最后进行。回顾到

$$
\xi_g^\chi = \sum_{\mu \nu} D_{\mu \nu} \xi_{g \mu \nu}^\chi
$$

留意我们是要对单个原子作偏导；考虑到 $(\mu, \nu)$ 的对称性，下式有两倍，但偏导只作用在 $\mu$ 上。由于公式表达比较困难，我们只能在这里用文字描述限制偏导的行为。

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial A_s} \leftarrow 2 \sum_{g \chi \mu \nu} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} D_{\mu \nu} \delta_{\mu \in A} \quad \text{(restrict $\partial$ to $\mu$)}
$$

我们会注意到，上式中的 $\nu$ 并不参与偏导，因此提前对 $\nu$ 作边际化 (就是前面 `fxc` 计算时的中间量 `ao_dm0` $\bar{\phi}_{g \mu}^{*}$)，可以将原先 `gu, gv -> uv` 的问题化为 `gu, gu -> u` 的问题，节省一次较大的矩阵乘法计算。

同时，该导数关于 $t, s$ 是对称的。因此，原先 $3 \times 3$ 的问题可以化为 $(xx, xy, xz, yy, yz, zz)$ 的 6 分量问题，稍微节省一些计算量。

因此，我们将会给出一个中间量 `dao_vxc_diag` $\mathscr{T}_{\mu}^{(ts)}$ (维度 $((ts), \mu)$，大小 $(6, n_\mathrm{AO})$)：

$$
\mathscr{T}_{\mu}^{(ts)} = 2 \sum_{g \chi \nu} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} D_{\mu \nu} \quad \text{(restrict $\partial$ to $\mu$)}
$$

随后引入原子依赖的求和计算：

$$
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial A_s} \leftarrow \sum_\mu \mathscr{T}_{\mu}^{(ts)} \delta_{\mu \in A}
$$

很显然上面一步是没有什么计算量的。至此，我们将问题化归为如何求取 `dao_vxc_diag` $\mathscr{T}_{\mu}^{(ts)}$ 中间量。在计算复杂度分析上，它与 `fxc` 倒是非常相似：其最大的计算量在 `ao_dm0` $\bar{\phi}_{g \mu}$ 的计算上，剩下的是大量复杂的 $O(N^2)$ 缩并。

在继续讨论前，我们指出，$w_g f_g^\chi$ 总是成对出现的。因此，我们可以先将 $w_g$ 与 $f_g^\chi$ 相乘，存储到 `wv` 或 `wvxc` 变量中。

下面我们对 LDA (RHO), GGA (SIGMA), MGGA (TAU) 分别给出 `dao_vxc_diag` 的具体公式。我们注意，对于 GGA 与 MGGA 任务，下述贡献项都需要叠加在一起，即 GGA 需要 LDA + SIGMA，MGGA 需要 LDA + SIGMA + TAU。

### 4.3 LDA (RHO)

LDA 部分对应 $\chi = \rho$，$\xi_{g \mu \nu}^{\chi = \rho} = \phi_{g \mu} \phi_{g \nu}$。仅对 $\mu$ 作偏导，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \rho}}{\partial t \partial s} = \phi_{g \mu}^{ts} \phi_{g \nu} \quad \text{(restrict $\partial$ to $\mu$)}
$$

代入 $\mathscr{T}_\mu^{(ts)}$ 的定义，并将 $\nu$ 缩并到 `ao_dm0` $\bar{\phi}_{g \mu} = \sum_\nu \phi_{g \nu} D_{\mu \nu}$ 上：

$$
\mathscr{T}_{\mu}^{(ts)} \mathrel{+}= 2 \sum_g w_g f_g^{\rho} \phi_{g \mu}^{ts} \bar{\phi}_{g \mu}
$$

程序上，$\phi_{g \mu}^{ts}$ 是 `ao` 张量在 $ts \in \{xx, xy, xz, yy, yz, zz\}$ 上的 6 个分量；$w_g f_g^\rho$ 即 `wv[0]`。该项是 $O(n_\mathrm{AO} n_\mathrm{grid})$ 复杂度。

### 4.4 GGA (SIGMA)

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

整体也是 $O(n_\mathrm{AO} n_\mathrm{grid})$ 复杂度。

### 4.5 MGGA (TAU)

MGGA (TAU) 部分对应 $\chi = \tau$，$\xi_{g \mu \nu}^{\chi = \tau} = \frac{1}{2} \sum_r \phi_{g \mu}^r \phi_{g \nu}^r$。仅对 $\mu$ 作偏导，

$$
\frac{\partial^2 \xi_{g \mu \nu}^{\chi = \tau}}{\partial t \partial s} = \frac{1}{2} \sum_r \phi_{g \mu}^{tsr} \phi_{g \nu}^r \quad \text{(restrict $\partial$ to $\mu$)}
$$

代入 $\mathscr{T}_\mu^{(ts)}$ 的定义，以 $\bar{\phi}_{g \mu}^r = \sum_\nu \phi_{g \nu}^r D_{\mu \nu}$ 缩并 $\nu$，并与 $\mathscr{T}_\mu^{(ts)}$ 定义中的 $2$ 系数相消：

$$
\mathscr{T}_{\mu}^{(ts)} \mathrel{+}= \sum_{g r} w_g f_g^\tau \phi_{g \mu}^{tsr} \bar{\phi}_{g \mu}^r
$$

程序上的对应与 GGA 第一项类似：对每个 $(ts)$ 分量，从 `ao` 中取出 $\phi_{g \mu}^{tsr}$ 共 3 个三阶导分量，与 `ao_dm0` 的 $r \in \{x, y, z\}$ 分量逐项配对，并以 `wv[4]` $w_g f_g^\tau$ 加权后缩并。同样是 $O(n_\mathrm{AO} n_\mathrm{grid})$ 复杂度。
