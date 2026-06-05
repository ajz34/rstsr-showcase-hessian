# RI-JK Skeleton 二阶导数详解

**LLM AI 生成提示**：该文档由 AI 编写，目前暂没有人工校对。如项目推进中遇到该文档存在问题，请联系维护者进行修正。

## 1. 积分记号系统

### 1.1 三中心 / 两中心积分

RI (Resolution of Identity) 方法中，库伦和交换积分通过辅助基组近似：

$$
(\mu\nu|\lambda\sigma) \approx (\mu\nu|P) (P|Q)^{-1} (Q|\lambda\sigma)
$$

其中：
- $(\mu\nu|P)$ = `int3c2e[μ, ν, P]`，形状 `[nao, nao, naux]`
- $(P|Q)$ = `int2c2e[P, Q]`，形状 `[naux, naux]`
- $(P|Q)^{-1}$ = `int2c2e_inv[P, Q]`，形状 `[naux, naux]`

### 1.2 导数积分命名

导数积分按导数作用的位置命名。数字 `ip` 表示对原子坐标求导的阶数：

**三中心积分** `(basis_μ | basis_ν | aux_P)`：

| 记号 | 积分名称 | 形状 | 含义 |
|------|---------|------|------|
| $(00\|0)$ | `int3c2e` | `[nao, nao, naux]` | 无导数 |
| $(10\|0)$ | `int3c2e_ip1` | `[3, nao, nao, naux]` | 对第一个基函数一阶导 |
| $(01\|0)$ | — | — | 对第二个基函数一阶导（通过对称性从 $(10\|0)$ 获得） |
| $(00\|1)$ | `int3c2e_ip2` | `[3, nao, nao, naux]` | 对辅助基函数一阶导 |
| $(11\|0)$ | `int3c2e_ipvip1` | `[3, 3, nao, nao, naux]` | 对两个基函数各求一阶导 |
| $(20\|0)$ | `int3c2e_ipip1` | `[3, 3, nao, nao, naux]` | 对第一个基函数二阶导 |
| $(10\|1)$ | `int3c2e_ip1ip2` | `[3, 3, nao, nao, naux]` | 对第一个基函数和辅助基各求一阶导 |
| $(00\|2)$ | `int3c2e_ipip2` | `[3, 3, nao, nao, naux]` | 对辅助基二阶导 |

**两中心积分** `(aux_P | aux_Q)`：

| 记号 | 积分名称 | 形状 | 含义 |
|------|---------|------|------|
| $(0\|0)$ | `int2c2e` | `[naux, naux]` | 无导数 |
| $(1\|0)$ | `int2c2e_ip1` | `[3, naux, naux]` | 对第一个辅助基一阶导 |
| $(1\|1)$ | `int2c2e_ip1ip2` | `[3, 3, naux, naux]` | 对两个辅助基各求一阶导 |
| $(2\|0)$ | `int2c2e_ipip1` | `[3, 3, naux, naux]` | 对第一个辅助基二阶导 |

### 1.3 贡献项命名

每个子贡献项以 `(basis_deriv | aux_deriv)(int2c2e_deriv)(basis_deriv | aux_deriv)` 形式描述。例如：

- `(10|0)(0|10)` 表示：左边三中心积分对基组一阶导 × 两中心积分无导数 × 右边三中心积分对基组一阶导
- `(00|1)(1|0)(0|00)` 表示：三中心积分为 `(00|1)` × 两中心积分为 `(1|0)` × 右边三中心积分为 `(00|0)`

## 2. 贡献项分解符号约定

### 2.1 einsum 中的指标

| 指标 | 含义 | 范围 |
|------|------|------|
| `t, s` | 笛卡尔坐标导数方向 | 3 |
| `u, v, k, l` | 原子轨道基函数 | nao |
| `P, Q, R, S, T` | 辅助基函数 | naux |
| `i, j` | 占据分子轨道 | nocc |
| `p, q` | 全部分子轨道 | nmo |

### 2.2 原子贡献的提取

skeleton 二阶导数的 Hessian 贡献具有 `[natm, natm, 3, 3]` 的形状。其提取方式为：

- 对基函数指标 `u`（对应原子 A 的基组片 `p0A:p1A`）和 `k`（对应原子 B 的基组片 `p0B:p1B`）进行切片求和
- 对辅助基指标 `P`（对应原子 B 的辅助基片 `p0B:p1B`）进行切片求和
- 对角块 `A=A` 的项：只有 `u` 切片到 A 的基组
- 非对角块 `A≠B`：`u` 切片到 A，`k` 切片到 B

对于辅助基索引的原子贡献，还需要做转置对称化：`de[A,B] += de[A,B] + de[A,B].transpose(1,0,3,2)`

## 3. RI-J Skeleton 导数

### 3.1 J20 — 基组二阶导数

J 的 RI 表达式为 $J = (\mu\nu|P)(P|Q)^{-1}(Q|\lambda\sigma) D_{\mu\nu} D_{\lambda\sigma}$，对基组求二阶导数：

| 子项 | 公式（einsum） | 因子 | 原子贡献 |
|------|---------------|------|---------|
| J20_1 | `tuvP, PQ, sklQ, uv, kl -> tsuk` | 4 | A(u), B(k) |
| J20_2 | `tsuvP, PQ, klQ, kl -> tsuv` | 2 | A(u), B(v) |
| J20_3 | `tsuvP, PQ, klQ, kl -> tsuv` | 2 | A=A(u=v) |

**推导说明**：

- **J20_1** `$(10|0)(0|10)$`：左边对第一个基函数一阶导 $(t\partial_\mu)$，右边对第一个基函数一阶导 $(s\partial_\lambda)$。对 $D_{\mu\nu} D_{\lambda\sigma}$ 缩并后得 `tsuk`，再按原子 A（u 切片）和 B（k 切片）提取。因子 4 来自 RHF 密度矩阵的 2 倍 × 两个密度矩阵交叉缩并 = $2 \times 2 = 4$。
  
- **J20_2** `$(11|0)(0|00)$`：左边对两个基函数各求一阶导 $(t\partial_\mu s\partial_\nu)$。缩并形式为 `tsuv`，需要再与 $D_{\mu\nu}$ 缩并。因子 2 来自一阶导数同时作用于 bra 和 ket。

- **J20_3** `$(20|0)(0|00)$`：左边对第一个基函数二阶导 $(t\partial s\partial_\mu)$。只有 A=B 时有贡献（二阶导数作用在同一原子的基组上）。

### 3.2 J11 — 基组一阶 × 辅助基一阶

| 子项 | 公式（einsum） | 因子 | 原子贡献 | 对称化 |
|------|---------------|------|---------|--------|
| J11_1 | `tsuvP, PQ, klQ, uv, kl -> tsuP` | 2 | A(u), B(P) | transpose |
| J11_2 | `tuvP, PQ, sQR, RS, klS, uv, kl -> tsuR` | 2 | A(u), B(R) | transpose |
| J11_3 | `tuvP, PQ, sQR, RS, klS, uv, kl -> tsuQ` | -2 | A(u), B(Q) | transpose |
| J11_4 | `tuvP, PQ, sklQ, uv, kl -> tsuQ` | 2 | A(u), B(Q) | transpose |

**推导说明**：

辅助基一阶导数的引入使得我们需要考虑 $(P|Q)^{-1}$ 对辅助基坐标的导数。由矩阵求逆法则：

$$
\frac{\partial (P|Q)^{-1}}{\partial R_A} = -(P|Q)^{-1} \frac{\partial (Q|R)}{\partial R_A} (R|S)^{-1}
$$

这产生了两种效应：
- **J11_2**：$(0|1)$ 项，即辅助基导数出现在第二个两中心积分位置
- **J11_3**：$(1|0)$ 项，即辅助基导数出现在第一个两中心积分位置，符号为负（来自矩阵求逆的链式法则）

J11_1 来自三中心积分对辅助基的导数 $(10|1)$，J11_4 来自三中心积分对辅助基的导数 $(00|1)$。

### 3.3 J02 — 辅助基二阶导数

| 子项 | 公式（einsum） | 因子 | 原子贡献 | 对称化 |
|------|---------------|------|---------|--------|
| J02_1 | `tsuvP, PQ, klQ, uv, kl -> tsP` | 1 | A=A(P) | 无 |
| J02_2 | `uvP, PQ, tsQR, RS, klS, uv, kl -> tsQ` | -1 | A=A(Q) | 无 |
| J02_3a | `uvP, PQ, tsQR, RS, klS, uv, kl -> tsQR` | -0.5 | A(Q), B(R) | transpose |
| J02_3b | `uvP, PQ, tQR, RS, sST, TU, klU, uv, kl -> tsQT` | -0.5 | A(Q), B(T) | transpose |
| J02_4 | `tuvP, PQ, sQR, RS, klS, uv, kl -> tsPQ` | -1 | A(P), B(Q) | transpose |
| J02_5 | `tuvP, PQ, sklQ, uv, kl -> tsPQ` | 0.5 | A(P), B(Q) | transpose |
| J02_6 | `uvP, PQ, tRQ, RS, sST, TU, klU, uv, kl -> tsRS` | 0.5 | A(R), B(S) | transpose |
| J02_7 | `tuvP, PQ, sRQ, RS, klS, uv, kl -> tsPR` | -1 | A(P), B(R) | transpose |
| J02_8 | `uvP, PQ, tQR, RS, sST, TU, klU, uv, kl -> tsRT` | 1 | A(R), B(T) | transpose |

**推导说明**：

J02 是最复杂的部分，包含 9 个子项。其复杂性来自辅助基二阶导数的多种来源：

1. **直接导数**：三中心或两中心积分直接对辅助基求二阶导（J02_1, J02_2）
2. **交叉导数**：对两个不同辅助基各求一阶导（J02_3a = `(1|1)`，J02_3b = `(1|0)(0|1)`）
3. **链式求导**：由于 $(P|Q)^{-1}$ 对辅助基的导数引入额外的 $(P|Q)^{-1}$，而 $(P|Q)^{-1}$ 自身也需要对辅助基求导，形成多级链式结构

关于因子：
- J02_3a 和 J02_3b 合在一起对应 $(P|Q)$ 的混合偏导 $\partial^2_{AB} (P|Q)$ 的两种分解：直接二阶（ip1ip2）和两个一阶的乘积（ip1 × ip1）
- J02_4/5/6/7/8 涉及辅助基导数通过 $(P|Q)^{-1}$ 链式传播到三中心积分的各种组合

## 4. RI-K Skeleton 导数

### 4.1 K 与 J 的结构对应

K（交换积分）的 RI 表达式为：

$$
K_{\mu\lambda} = (\mu\nu|P)(P|Q)^{-1}(Q|\lambda\sigma) D_{\nu\sigma}
$$

与 J 的关键区别在于：K 的密度矩阵缩并是交叉的（$\nu$ 与 $\sigma$），而 J 是直积的（$\mu\nu$ 与 $\lambda\sigma$ 分别缩并）。这意味着 K 需要引入占据轨道系数来做缩并。

实际实现中，使用 `mocc_2 = mocc * sqrt(occ)` 来替代 `dm0`，即：

- J 中：`uv, kl` 缩并 → `dm0[u,v] * dm0[k,l]`
- K 中：`vi, li, uj, kj` 缩并 → `mocc_2[v,i] * mocc_2[l,i] * mocc_2[u,j] * mocc_2[k,j]`

### 4.2 K20

| 子项 | 公式（einsum） | 因子 | 原子贡献 |
|------|---------------|------|---------|
| K20_1a | `tuvP, PQ, sklQ, ui, vj, ki, lj -> tsuk` | 2 | A(u), B(k) |
| K20_1b | `tuvP, PQ, sklQ, ui, vj, kj, li -> tsuk` | 2 | A(u), B(k) |
| K20_2 | `tsuvP, PQ, klQ, ui, vj, ki, lj -> tsuv` | 2 | A(u), B(v) |
| K20_3 | `tsuvP, PQ, klQ, ui, vj, ki, lj -> tsuv` | 2 | A=A(u=v) |

**K20_1 的拆分**：K20_1 对应 J20_1 的 `(10|0)(0|10)` 项。但由于 K 的密度矩阵缩并是交叉的，交换积分有两部分贡献：
- **1a**：`ki, lj` 缩并（即 $\delta_{ki} \delta_{lj}$，对应 $K_{\mu\lambda}$ 的直接项）
- **1b**：`kj, li` 缩并（即交换 $i \leftrightarrow j$，对应 $K_{\mu\lambda}$ 的交换项）

这两部分在 J 中因为直积结构自动合并为 4 倍因子，在 K 中需要分开计算。

### 4.3 K11 和 K02

K11 和 K02 的子项结构与 J11/J02 完全对应，唯一区别是将 `dm0` 的缩并替换为 `mocc_2` 的四指标缩并。因子也相同。

K02 的 einsum 注意事项：
- K02_8 的 einsum 中，最后一个缩并指标在不同实现中可能为 `QS` 或 `RT`，这取决于具体的辅助基索引命名，但数学上等价

## 5. hcore 和 ovlp 的 skeleton 导数

### 5.1 核排斥 Hessian

纯几何量，不依赖密度矩阵。公式为标准的点电荷库伦二阶导数：

$$
\frac{\partial^2 E_{\text{nuc}}}{\partial R_A^t \partial R_B^s} = \frac{Z_A Z_B}{|R_A - R_B|^3}\delta_{ts} - \frac{3 Z_A Z_B (R_A - R_B)_t (R_A - R_B)_s}{|R_A - R_B|^5}
$$

对角块需减去其他所有原子的贡献之和。

### 5.2 hcore skeleton 二阶导数

核心哈密顿 $H_{\mu\nu} = T_{\mu\nu} + V_{\mu\nu}^{\text{nuc}}$ 的二阶导数（skeleton）通过以下积分构造：

- `int1e_ipipkin` + `int1e_ipkinip`：动能的二阶导数（aa 型和 ab 型）
- `int1e_ipipnuc` + `int1e_ipnucip`：核吸引势的二阶导数
- `int1e_ipiprinv` + `int1e_iprinvip`：rinv 积分（用于消除特定原子的核吸引势贡献）
- ECP 相关积分（如存在）

关键处理：
- 对角块 `A=B`：需要添加 rinv 积分来补偿核吸引势在 A 原子处的不连续性
- 非对角块 `A≠B`：需要分别处理 rinv@A 作用于 B 的基组、以及 rinv@B 作用于 A 的基组
- 最终结果需要加上自身的转置：`hcore_deriv += hcore_deriv.swapaxes(-1, -2)`

### 5.3 overlap 导数贡献

重叠积分导数的贡献 **不是** skeleton 导数。它来自 Hellmann-Feynman 定理，将密度矩阵响应中与 $S'$ 相关的部分转化为与 $D_e$（能量加权密度矩阵）的直接缩并：

$$
E_{\text{ovlp}}[A,B] = -2 \text{tr}(D_e \cdot S''[A,B])
$$

其中 $S''$ 使用 `int1e_ipipovlp`（aa 型，A=B）和 `int1e_ipovlpip`（ab 型，A≠B）。
