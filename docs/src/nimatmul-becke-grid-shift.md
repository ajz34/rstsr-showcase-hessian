# DFT 基于 Becke 配分的格点偏移导数

该文档将解决偏移导数问题。

## 1. 格点偏移导数概述

### 1.1 格点坐标权重与原子坐标的关系

回顾到对于任意关于电子坐标的函数 $F(\bm{r})$，数值格点积分表达式是

$$
E = \int F(\bm{r}) \, d\bm{r} \simeq \sum_{g} w_g F(\bm{r}_g)
$$

我们现在需要一点一点增加复杂性。在梯度问题中，我们要注意到，函数 $F$ 经常还同时是关于原子坐标 $\bm{R}_A$ 的函数。于是我们有

$$
E(\{\bm{R}_A\}) = \int F(\bm{r}; \{\bm{R}_A\}) \, d\bm{r} \simeq \sum_{g} w_g F(\bm{r}_g; \{\bm{R}_A\})
$$

依该近似表达式求导，可以得到

$$
\frac{\mathrm{d} E}{\mathrm{d} \bm{R}_A} \simeq \sum_{g} w_g \frac{\partial F}{\partial \bm{R}_A}
$$

在格点 $g$ 完备的情况下，上式是精确的。这也是为什么通常的杂化 GGA 梯度计算中，格点偏移导数可以忽略的原因。

但现实中，格点数目是有限的；并且 **格点坐标 $\bm{r}_g$ 事实上依赖于原子坐标 $\{ \bm{R}_A \}$** (格点坐标是由原子为中心展开的 Lebedev 球面格点所构成的)。同时，**格点权重 $w_g$ 也依赖于原子坐标 $\{ \bm{R}_A \}$** (Lebedev 格点权重本身是固定的，但其权重还依 Becke partition 缩放)。因此我们有

$$
E(\{\bm{R}_A\}) \simeq \sum_{g} w_g(\{\bm{R}_A\}) F(\bm{r}_g(\{\bm{R}_A\}); \{\bm{R}_A\})
$$

对该表达式求导，得到

$$
\frac{\mathrm{d} E}{\mathrm{d} \bm{R}_A} \simeq \sum_{g} \left( w_g \frac{\partial F}{\partial \bm{R}_A} + \frac{\partial w_g}{\partial \bm{R}_A} F + w_g \frac{\partial F}{\partial \bm{r}_g} \frac{\partial \bm{r}_g}{\partial \bm{R}_A} \right)
$$

相比于之前的近似表达式，这里多了两项：一项是格点权重的导数，另一项是格点坐标的导数。我们统称这两项为 **格点偏移导数**。

> **格点偏移导数包含两个非常不同的贡献，而非仅格点权重**
>
> 这是我个人很长时间遇到的误解；即格点偏移导致的梯度只包含格点权重导数贡献。这种误解来源于格点权重显含在公式中，而格点坐标相对于原子坐标的依赖关系是隐含的。

在通常的 Lebedev 导出的 DFT 格点中，格点 $g$ 总是由其对应的特定原子生成的。如果格点 $g$ 是由原子 $A$ 生成的，则 

$$
\frac{\partial \bm{r}_g}{\partial \bm{R}_B} = \delta_{AB} \bm{I}, \quad g \in A
$$

这里通过一阶梯度引入了格点偏移导数的概念；我们会在后续小节再讨论二阶梯度的格点偏移导数。

### 1.2 验证手段：对 Grad/Hess 的原子指标求和

> **验证手段是必要条件而非充分条件**
>
> 如果格点偏移导数实现正确，则对 Grad/Hess 的原子指标求和应该为零；但反之不成立。

格点偏移导数本身经常很小：对于 GGA 一阶梯度问题，它引起的误差经常在 10<sup>-6</sup> Hartree/Bohr 级别；但对于 MGGA 二阶梯度问题，它又经常引起数十到数百 cm<sup>-1</sup> 的频率误差。如何验证其正确是一件非常微妙的事情。

针对一阶梯度，我们会引入平移不变性验证。具体来说，记能量是原子坐标的函数 $E(\{\bm{R}_A\})$；如果我们对所有原子坐标进行平移 $\bm{R}_A \to \bm{R}_A + \bm{\epsilon}$，则能量应该不变。于是我们有

$$
\frac{\partial E (\bm{R}_A + \bm{\epsilon}, \bm{R}_B + \bm{\epsilon}, \ldots)}{\partial \bm{\epsilon}} = \frac{\partial E}{\partial \bm{R}_A} + \frac{\partial E}{\partial \bm{R}_B} + \cdots = 0
$$

该等式对于所有取向的 $\bm{\epsilon}$ 都成立，那么如果我们将 $\bm{\epsilon}$ 设置为 $t \in \{ x, y, z \}$，则可以得到三个独立的方程。下述方程即是梯度对原子指标求和的验证手段：

$$
\sum_A \frac{\partial E}{\partial R_{At}} = 0, \quad t \in \{ x, y, z \}
$$

对于二阶梯度同理：

$$
\sum_{A B} \frac{\partial^2 E}{\partial R_{At} \partial R_{Bs}} = 0, \quad t, s \in \{ x, y, z \}
$$

在实际程序实现中肯定会有机器误差；但这部分误差在一阶梯度一般不会超过 1e-13 Hartree/Bohr，在二阶梯度一般不会超过 1e-9 Hartree/Bohr<sup>2</sup>。

上述验证手段不仅可以用于普通的 Hessian，也可以应用于 Skeleton 导数专有的贡献、以及 Fock 矩阵导数。

## 2. DFT 能量格点偏移一阶 Skeleton 梯度回顾

我们现在用梯度公式中比较易用的符号来说明问题。这里的记号与前一节会有不同，特别是 $R_{At}$ 会变更为 $A_t$。

我们以程序推导所用的记号，重新标明一阶 Skeleton 梯度回顾。

$$
\partial_{A_t} E = \sum_g w_g \frac{\partial (f \rho)_g}{\partial A_t} + \sum_g (\partial_{A_t} w_g) f_g \rho_g + \sum_g w_g \frac{\partial (f \rho)_g}{\partial t} \delta_{g \in A}
$$

- 第 1 项是常规的梯度贡献；这也是通常梯度程序所需要实现的部分。
- 第 2 项是格点权重偏移导数贡献；如果我们已经获得了 $\partial_{A_t} w_g$，则该项的实现是非常直接的。
- 第 3 项是格点坐标偏移导数贡献。

这里我们有必要讨论第 1 项与第 3 项的联系。

我们首先了解第 1 项。我们文档主要讨论二阶梯度，之前没有引入一阶梯度的推演；这里作简单展示。其程序实现大致推导为：

$$
\begin{align*}
\partial_{A_t} E
&\leftarrow \sum_{g \chi} w_g \frac{\partial (f \rho)_g}{\partial \xi^\chi_g} \frac{\partial \xi^\chi_g}{\partial A_t}
= \sum_{g \chi} w_g f^\chi_g \frac{\partial \xi^\chi_g}{\partial A_t}
\\
&= \sum_{g \chi} \sum_{\mu \nu} \left( w_g f^\chi_g \frac{\partial \xi_{g \mu \nu}^\chi}{\partial t} D_{\mu \nu} \times - \delta_{\mu \in A} + \mathrm{swap}(\mu, \nu) \right)
\quad (\text{restrict $\partial$ to $\mu$})
\end{align*}
$$

我们交换求和顺序，引入 2-dim 临时张量 $\mathscr{T}_{\mu}^t$：

$$
\mathscr{T}_{\mu}^t = \sum_{g \chi} w_g f^\chi_g \frac{\partial \xi_{g \mu \nu}^\chi}{\partial t} D_{\mu \nu}
$$

那么一阶 Skeleton 梯度的普通项导数可以写为 (2 倍来源于 $\mathrm{swap}(\mu, \nu)$)：

$$
\partial_{A_t} E \leftarrow - 2 \sum_{\mu \in A} \mathscr{T}_{\mu}^t
$$

如何实现 $\mathscr{T}_{\mu}^t$ 会是整个程序的最大难点；但在格点偏移问题的讨论中，这不是最大的重点。我们现在了解第 3 项即格点坐标偏移导数贡献的表达式：

$$
\begin{align*}
\partial_{A_t} E &\leftarrow \sum_{g \chi} w_g f^\chi_g \frac{\partial \xi_g^\chi}{\partial t} \delta_{g \in A} \\
\\
&= \sum_{g \chi} \sum_{\mu \nu} w_g f_g^\chi \frac{\partial \xi_{g \mu \nu}^\chi}{\partial t} D_{\mu \nu} \delta_{g \in A} + \mathrm{swap}(\mu, \nu)
\quad (\text{restrict $\partial$ to $\mu$})
\end{align*}
$$

这里有两种计算策略：
- 充分解耦地计算 $\partial_t \xi_g^\chi$ 后作简单缩并；
- 不解耦 $\partial_t \xi_{g \mu \nu}^\chi$，作比较复杂的缩并。

这两个策略在二阶梯度程序中都会出现；但在一阶梯度程序中，第二种策略一般是更好的选择：
- 充分解耦计算 $\partial_t \xi_g^\chi$ 这件事本身就是代价。它本身的意义很清晰，但其计算复杂程度与计算量并不亚于 $\mathscr{T}_{\mu}^t$。

但这不意味着先前的程序就能无痛地直接使用。我们需要对 $\mathscr{T}_{\mu}^t$ 的计算过程进行修改，在公式表达上引入 3-dim 张量 $\mathscr{T}_{\mu}^{A t}$：

$$
\mathscr{T}_{\mu}^{A t} = \sum_{g \chi} w_g f^\chi_g \frac{\partial \xi_{g \mu \nu}^\chi}{\partial t} D_{\mu \nu} \delta_{g \in A}
$$

它其实仅仅就是在格点求和时，依据格点 $g$ 是由哪个原子 $A$ 所生成的 Lebedev 格点，来分割求和结果。这个过程并不复杂，也并不难以做到；但它必须要求下述数据结构要求：

> **实现决策：格点优先依原子排序**
>
> 决策依据：计算 $\mathscr{T}_{\mu}^{A t}$ 时涉及到矩阵乘法，其被乘的指标是格点 $g$。该指标只有在保证较好的连续性时才能获得可观的性能。
>
> 代价：我们目前仍然是基于 matmul 实现的 DFT 积分程序；但当分子或固体体系很大时，更合适的做法是依格点与原子轨道中心的距离作一定程度的稀疏化。稀疏化对连续性的要求是格点 $g$ 所对应的空间坐标 $\bm{r}_g$ 相近，而不是 $g$ 是否由同一个原子的 Lebedev 格点生成。我们需要注意到 Lebedev 格点经常会取用比较大的半径 (最大可能有 10 Angstrom)，因此不同原子生成的 Lebedev 格点可能会有重叠。
> 
> 未来的一种可能解决方案是，在依原子预先归类后，再进行基于空间坐标的排序 (有些接近于桶排序组合)。这不算太 trivial，因为一般来说生成 DFT 格点的最简单做法是先依原子迭代、后依半径迭代，最后用 Lebedev 打表数据给出；这显然有少许空间排序上的优化空间。但这不是当前文档与程序的实现目标。

在具体的程序实现中，我们总是对格点作分批的。因此原先用于计算 $\mathscr{T}_{\mu}^t$ 的程序是能复用的，只是在得到一批 (必须是同一原子生成的 Lebedev) 格点的 $\mathscr{T}_{\mu}^{A t}$ 增量后，计算进第 1 项普通 Skeleton 导数贡献后，不着急者扔掉：

$$
\partial_{A_t} E \leftarrow - 2 \sum_{\mu \in A} \sum_B \mathscr{T}_{\mu}^{B t}
$$

(上式尽管从公式上，是对 $B$ 求和；但实际上一批格点一定只是由同一个原子生成的 Lebedev 格点，因此具体程序中体现为总是增加这个 2-dim 张量)

我们还可以立即复用这个张量，以实现第 3 项格点坐标偏移 Skeleton 导数贡献：

$$
\partial_{A_t} E \leftarrow 2 \sum_\mu \mathscr{T}_{\mu}^{A t}
$$

## 3. DFT 能量格点偏移二阶 Skeleton 梯度实现

### 3.1 格点密度导数及其应用情景

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `drho` | $\partial_{A_t} \xi_g^\chi$ | $(g, \chi, t, A)$</br>`[g, x, t, A]` | `[ngrids, nvar, 3, natm]` | |
| `prho` | $\partial_{t} \xi_g^\chi$ | $(g, \chi, t)$</br>`[g, x, t]` | `[ngrids, nvar, 3]` | |

我们回顾到 $\partial_{A_t} = - \partial_t \delta_{\mu \in A}$，那么对 $\partial_{A_t}$ 原子指标 $A$ 求和可以立即得到关于 $\partial_t$ 的关系：

$$
\partial_t \xi_g^\chi = - \sum_A \partial_{A_t} \xi_g^\chi
$$

```rust
let prho = drho.sum_axes(3);
```

由于这部分计算几乎没有代价，且 `drho` $\partial_{A_t} \xi_g^\chi$ 是二阶梯度程序中必须计算的，因此我们可以立即复用它来计算 $\partial_t \xi_g^\chi$，以简化一部分格点坐标偏移 Skeleton 导数。

但也需要指出，整个二阶梯度程序是不计算 $\partial_{A_t} \partial_{B_s} \xi_g^\chi$ 即格点密度二阶导数的：这一方面计算量太大，另一方面这是相当大的 $O(N^3)$ 内存需求。因此，如果碰到二阶密度导数的需求，必须要作密度在原子轨道上的展开，用更复杂 (但可复用普通二阶梯度中间量) 的方式计算。

### 3.2 格点偏移二阶 Skeleton 导数概述

我们这里展示二阶梯度的完整表达式：

$$
\begin{alignat*}{4}
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s}
&= \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^\chi}{\partial A_t} \frac{\partial \xi_g^{\chi'}}{\partial B_s}
&\quad& \texttt{de\_fxc} \\
&+ \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial A_t \partial B_s}
&\quad& \texttt{de\_vxc} \\
&+ \sum_{g \chi} \frac{\partial w_g}{\partial A_t} f_g^\chi \frac{\partial \xi_g^\chi}{\partial B_s} + \mathrm{swap}(A_t, B_s)
&\quad& \texttt{de\_becke\_full\_1} \\
&+ \sum_{g} \frac{\partial^2 w_g}{\partial A_t \partial B_s} f_g \rho_g
&\quad& \texttt{de\_becke\_full\_2} \\
&+ \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^\chi}{\partial t} \frac{\partial \xi_g^{\chi'}}{\partial B_s} \delta_{g \in A} + \mathrm{swap}(A_t, B_s)
&\quad& \texttt{de\_becke\_atom\_1} \\
&+ \sum_{g \chi} \frac{\partial w_g}{\partial B_s} f_g^\chi \frac{\partial \xi_g^\chi}{\partial t} \delta_{g \in A} + \mathrm{swap}(A_t, B_s)
&\quad& \texttt{de\_becke\_atom\_2} \\
&+ \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^\chi}{\partial t} \frac{\partial \xi_g^{\chi'}}{\partial s} \delta_{g \in A}
&\quad& \texttt{de\_becke\_atom\_3} \\
&+ \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial t \, \partial B_s} \delta_{g \in A} + \mathrm{swap}(A_t, B_s)
&\quad& \\
&+ \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial t \partial s} \delta_{g \in A} \delta_{A B}
&\quad& \texttt{de\_becke\_vxc}
\end{alignat*}
$$

这其中，
- `de_fxc`, `de_vxc` 是普通的二阶梯度贡献。
- `de_becke_full_1`, `de_becke_atom_1`, `de_becke_atom_2`, `de_becke_atom_3` 四项是可以基于普通二阶梯度的中间量 (以及 Becke partition 所给出的格点权重梯度)，立即使用 einsum 计算的。在 REST 正式程序中不推荐使用 einsum，这些项会在实际程序中通过多步 vecdot 实现。
- `de_becke_full_2` 这一项比较特殊。它涉及到权重二阶梯度 $\partial_{A_t} \partial_{B_s} w_g$，若要存储下来是 $O(N^3)$。即使要对格点分批，它也会占用较大的 memory footprint。我们在实际程序中，会作 fused-contraction：提前给定被缩并张量 $f_g \rho_g$，返回一个维度大小 $(A, B, t, s)$ 的 Hessian 增量。
- `de_becke_vxc` 这一项 (两行) 是涉及到密度格点二阶导数的部分；我们会避免直接生成这类张量。这意味着我们需要复用其中的中间张量计算过程。
- 最后一行的 $\delta_{AB}$ 来源是，这里格点导数同时要求 $\delta_{g \in A}$ 与 $\delta_{g \in B}$，因此只有当 $A = B$ 时才有贡献。后文的公式推演上，用 $\delta_{AB}$ 比较方便。

### 3.3 `de_becke_vxc` 贡献程序实现

`de_becke_vxc` 是八个贡献中唯一涉及密度格点二阶导数的项，它包含混合二阶 $\partial_t \partial_{B_s} \xi_g^\chi$ 与纯空间二阶 $\partial_t \partial_s \xi_g^\chi$。这两项计算代价会是非常大的 $O(N^3)$，理想的情况下是不直接生成它们，而是复用普通二阶梯度中间量 (c.f. [普通二阶 Skeleton 梯度](nimatmul-skeleton2) 文档的 4.2 小节 `dao_vxc_diag` 与 5.1 小节 `dao_vxc_off`) 来计算。

我们先回顾 `dao_vxc_diag` 与 `dao_vxc_off` 中间量。与先前记号不同的是，我们现在引入特定原子上格点的求和：

$$
\begin{alignat*}{4}
\mathscr{T}_{\mu}^{A(ts)} &= 2 \sum_{g \in A} \sum_{\chi \nu} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} D_{\mu \nu}
&\quad& \text{(restrict $\partial$ to $\mu$, \texttt{dao\_vxc\_diag})} \\
\mathscr{T}_{\mu \nu}^{A t s} &= \sum_{g \in A} \sum_{\chi} w_g f_g^\chi \frac{\partial^2 \xi_{g \mu \nu}^\chi}{\partial t \partial s} + \text{swap} (t \mu, s \nu)
&\quad& \text{(restrict $\partial_t$ to $\mu$, $\partial_s$ to $\nu$, \texttt{dao\_vxc\_off})}
\end{alignat*}
$$

这两项临时张量与最终的二阶 Skeleton 梯度之间的关系是

$$
\begin{align*}
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s}
&\leftarrow \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial t \, \partial B_s} \delta_{g \in A} + \mathrm{swap} (A_t, B_s) \\
&= - \sum_{\mu \in B} \Big( \mathscr{T}_{\mu}^{A(ts)} + \sum_\nu \mathscr{T}_{\mu \nu}^{A st} D_{\mu \nu} \Big) + \mathrm{swap} (A_t, B_s) \\
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s}
&\leftarrow \sum_{g \chi} w_g f_g^\chi \frac{\partial^2 \xi_g^\chi}{\partial t \, \partial s} \delta_{g \in A} \delta_{A B} \\
&= \frac{1}{2} \delta_{A B} \sum_{\mu} \Big( \mathscr{T}_{\mu}^{A(ts)} + \sum_\nu \mathscr{T}_{\mu \nu}^{A ts} D_{\mu \nu} \Big) + \mathrm{swap} (A_t, B_s)
\end{align*}
$$

我们刻意在 $\partial_t \partial_s \xi_g^\chi$ 一项中引入 $\frac{1}{2}$ 系数，单纯是为了与 $\partial_t \partial_{B_s} \xi_g^\chi$ 所用到的 $\mathrm{swap} (A_t, B_s)$ 相匹配。

由此，我们重组 `de_becke_vxc` 的计算过程为：

$$
\begin{alignat*}{4}
\frac{\partial^2 E^\text{xc}}{\partial A_t \partial B_s}
&\leftarrow \frac{1}{2} \delta_{AB} \sum_{\mu} \mathscr{T}_{\mu}^{A(ts)} - \sum_{\mu \in B} \mathscr{T}_{\mu}^{A(ts)}
&\quad& \texttt{de\_becke\_vxc\_diag} \\
&+ \frac{1}{2} \delta_{AB} \sum_{\mu \nu} \mathscr{T}_{\mu \nu}^{A ts} D_{\mu \nu} - \sum_{\mu \in B} \sum_\nu \mathscr{T}_{\mu \nu}^{A st} D_{\mu \nu}
&\quad& \texttt{de\_becke\_vxc\_off} \\
&+ \mathrm{swap} (A_t, B_s)
\end{alignat*}
$$

**函数 `get_de_becke_vxc_parts`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 |
|--|--|--|--|
| `dao_vxc_diag` | $\mathscr{T}_{\mu}^{A(ts)}$</br>$A$ specified | $(\mu, (ts))$</br>`[u, ts]` | `[nao, 6]` |
| `dao_vxc_off` | $\mathscr{T}_{\mu \nu}^{A ts}$</br>$A$ specified | $(\mu, \nu, t, s)$</br>`[u, v, 3, 3]` | `[nao, nao, 3, 3]` |
| `dm0` | $D_{\mu \nu}$ | $(\mu, \nu)$</br>`[u, v]` | `[nao, nao]` | |
| `atm_idx` | current chunk</br>of atom $A$ | | |
| `aoslices` | | | `natm` |
| `de_becke_vxc_diag`</br>(increment output) | $A$ specified | $(t, s, B)$</br>`[t, s, B]` | `[3, 3, natm]` |
| `de_becke_vxc_off`</br>(increment output) | $A$ specified | $(t, s, B)$</br>`[t, s, B]` | `[3, 3, natm]` |

请留意，程序实现中有一些公式难以表达清楚的部分：
- 该函数是在格点 $g$ 的分批下完成的；这一批格点一定对应于同一个原子 $A$ 的 Lebedev 格点。因此，这里的所有 $A$ 指标都没有出现在维度大小中。
- 尽管程序会输出 `de_becke_vxc_diag/off`，但它们是没有经过 $\mathrm{swap} (A_t, B_s)$ 对称化的。对称化过程会在格点 chunk 分批的过程中，拼接到完整的 $(t, s, A, B)$ 维度的 Hessian 中完成。

**函数 `contract_pvxc`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 |
|--|--|--|--|
| `pvxc` | $\mathscr{T}_\mu^{A (t s)}$ (diag)</br>$\sum_\nu \mathscr{T}_{\mu \nu}^{A ts} D_{\mu \nu}$ (off) | $(\mu, t, s)$</br>`[u, t, s]` | `[nao, 3, 3]` |
| `atm_idx` | $A$ | | |
| `aoslices` | | | `natm` |
| `de_pvxc`</br>(output) | | $(t, s, B)$</br>`[t, s, B]` | `[3, 3, natm]` |

求梯度的过程中有一个完全相同的计算结构；函数 `contract_pvxc` 就是用来处理该结构的：

$$
\mathscr{P}_{t s}^B = \frac{1}{2} \delta_{AB} \sum_{\mu} \mathscr{S}_\mu^{t s} - \sum_{\mu \in B} \mathscr{S}_\mu^{t s}
$$

针对 diag/off 两种情况，这里的 $\mathscr{S}_\mu^{t s}$ 分别对应于

$$
\mathscr{S}_\mu^{t s} = \begin{cases}
\mathscr{T}_\mu^{A (t s)} \delta_{AB}, & \texttt{diag} \\
\sum_\nu \mathscr{T}_{\mu \nu}^{A ts} D_{\mu \nu}, & \texttt{off}
\end{cases}
$$

这里会有一个比较容易迷惑的点。这里的张量是 $\mathscr{P}_{ts}^B$；它从表面上不包含指标 $A$，但实际上它是基于原子 $A$ 所生成格点的 batch/chunk 计算，必须要服务于特定的原子 $A$。这也是为什么在 `contract_pvxc` 函数中，`atm_idx` 仍然保留在函数参数中。

## 4. DFT Fock 格点偏移一阶 Skeleton 梯度实现

### 4.1 格点偏移 Fock 一阶 Skeleton 梯度概述

这里展示 Fock 矩阵一阶梯度的完整表达式：

$$
\begin{alignat*}{4}
V_{\mu \nu}^\text{xc} &= \sum_{g \chi} w_g f_g^\chi \xi_{g \mu \nu}^\chi
\\
\frac{\partial V_{\mu \nu}^\text{xc}}{\partial A_t}
&= \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^{\chi'}}{\partial A_t} \xi_{g \mu \nu}^\chi
&\quad& \texttt{vmat\_fxc} \\
&+ \sum_{g \chi} w_g f_g^\chi \frac{\partial \xi_{g \mu \nu}^\chi}{\partial A_t}
&\quad& \texttt{vmat\_vxc} \\
&+ \sum_{g \chi} \frac{\partial w_g}{\partial A_t} f_g^\chi \xi_{g \mu \nu}^\chi
&\quad& \texttt{vmat\_becke\_dw} \\
&+ \sum_{g \chi \chi'} w_g f_g^{\chi \chi'} \frac{\partial \xi_g^{\chi'}}{\partial t} \xi_{g \mu \nu}^\chi \delta_{g \in A}
&\quad& \texttt{vmat\_becke\_fxc} \\
&+ \sum_{g \chi} w_g f_g^\chi \frac{\partial \xi_{g \mu \nu}^\chi}{\partial t} \delta_{g \in A}
&\quad& \texttt{vmat\_becke\_vxc}
\end{alignat*}
$$

其中 `vmat_fxc`, `vmat_vxc` 是普通的 Fock 一阶梯度贡献；`vmat_becke_dw`, `vmat_becke_fxc`, `vmat_becke_vxc` 是格点偏移导数贡献。

**函数 `get_vmat_becke_parts`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `xc_type` | | | `LDA` / `GGA` / `MGGA` | |
| `ao` | $\phi_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>4/10/10 |
| `vxc` | $f_g^\chi$ | $(g, \chi)$</br>`[g, x]` | `[ngrids, nvar]` | |
| `fxc` | $f_{g}^{\chi \chi'}$ | $(g, \chi, \chi')$</br>`[g, x, x']` | `[ngrids, nvar, nvar]` | |
| `prho` | $\partial_t \xi_g^\chi$ | $(g, \chi, t)$</br>`[g, x, t]` | `[ngrids, nvar, 3]` | |
| `w` | $w_g$ | $(g)$</br>`[g]` | `[ngrids]` | |
| `dw` | $\partial_{A_t} w_g$ | $(g, t, A)$</br>`[g, t, A]` | `[ngrids, 3, natm]` | |
| `vmat_ip` | $\mathscr{T}_{\mu \nu}^{t}$ | $(\mu, \nu, t)$</br>`[u, v, t]` | `[nao, nao, 3]` | |

这里不作细致展开。总地来说，
- `vmat_becke_dw` 与 `vmat_becke_fxc` 的实现方式与 Fock 矩阵计算一致，这在程序中使用 `xc_fock_stack` 实现。
- `vmat_becke_vxc` 实现用到关键中间量 `vmat_ip`，从而在普通 Fock 矩阵一阶 Skeleton 导数计算之外不需要任何额外计算量。

**函数 `xc_fock_stack`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `xc_type` | | | `LDA` / `GGA` / `MGGA` | |
| `ao` | $\phi_{g \mu}^{*}$ | $(g, \mu, *)$</br>`[g, u, *]` | `[ngrids, nao, ncomp]` | `ncomp`</br>4/10/10 |
| `wv` | | $(g, \chi, \mathbb{K})$</br>`[g, x, k]` | `[ngrids, nvar, nk]` |

该函数可以用于计算 Fock 矩阵的相关量 (`vmat_fxc`, `vmat_becke_dw`, `vmat_becke_fxc`)。它的实现方式与普通 Fock 矩阵计算一致，但允许 $\mathbb{K}$ 个，即引入导数分量的计算。