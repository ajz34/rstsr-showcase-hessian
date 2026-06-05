# CP-HF 与响应函数

**LLM AI 生成提示**：该文档由 AI 编写，目前暂没有人工校对。如项目推进中遇到该文档存在问题，请联系维护者进行修正。

## 1. 一阶导数构造

### 1.1 Fock 矩阵的一阶 skeleton 导数 `f1ao`

Fock 矩阵的一阶 skeleton 导数（不含密度矩阵响应）为：

$$
F_{\mu\nu}^{(1),A} = H_{\mu\nu}^{(1),A} + J_{\mu\nu}^{(1),A}[D] - \frac{1}{2} K_{\mu\nu}^{(1),A}[D]
$$

构造步骤：

1. **hcore 部分** (`generator_hcore_deriv1`)：
   - 由 `int1e_ipkin`、`int1e_ipnuc`、`int1e_iprinv` 等积分构造
   - 对每个原子 A，需要分别处理基组在 A 上的贡献和 rinv@A 的贡献
   - 最终对称化：`h1ao + h1ao.swapaxes(-1, -2)`

2. **J 部分** (`get_rij_deriv1_ao_naive`)：
   - 分为 `j1ao_aux0`（辅助基零阶）和 `j1ao_aux1`（辅助基一阶）
   - aux0 部分有 4 个贡献：`(10|0)(0|00)`、`(01|0)(0|00)`、`(00|0)(0|10)`、`(00|0)(0|01)`
   - aux1 部分有 4 个贡献：`(00|1)(0|00)`、`(00|0)(1|00)`、`(00|0)(1|0)(0|00)`、`(00|0)(0|1)(0|00)`

3. **K 部分** (`get_rik_deriv1_ao_naive`)：
   - 结构与 J 部分对称，但密度矩阵缩并为占据轨道交叉缩并

### 1.2 重叠矩阵的一阶导数 `s1ao`

$$
S_{\mu\nu}^{(1),A} = -\langle \chi_\mu^{(1),A} | \chi_\nu \rangle - \langle \chi_\mu | \chi_\nu^{(1),A} \rangle
$$

实现上使用 `int1e_ipovlp` 积分，并按原子 A 的基组切片：

```python
s1ao[:, slc, :] += -int1e_ipovlp[:, slc, :]
s1ao[:, :, slc] += -int1e_ipovlp[:, slc, :].swapaxes(-1, -2)
```

注意符号为负（来自 `ip` 积分的定义约定）。

### 1.3 MO 表象转换

为 CP-HF 准备：

```python
f1mo = mo_coeff.T @ f1ao @ mocc   # [natm, 3, nmo, nocc]
s1mo = mo_coeff.T @ s1ao @ mocc   # [natm, 3, nmo, nocc]
```

注意只对 ket 端转换到占据轨道（half-transformation），这是 CP-HF 的关键。

## 2. CP-HF 方程

### 2.1 标准形式

CP-HF 方程的标准形式为（基于 PySCF 的 `solve_withs1`）：

$$
(\varepsilon_a - \varepsilon_i) U_{ai}^A - A_{ai,bj} U_{bj}^A = -B_{ai}^A
$$

其中：
- $A_{ai,bj}$ 是响应算符（在 RHF 中等价于 Fock/veff 算符）
- $B_{ai}^A = F_{ai}^{(1),A} - S_{ai}^{(1),A} \varepsilon_i$
- $U_{ai}^A$ 是占据-虚轨道的响应系数

对于占据-占据块（用于处理 overlap 一阶导数），约定：

$$
U_{ji}^A = -\frac{1}{2} S_{ji}^{(1),A}
$$

### 2.2 Dimensionless 形式（pyhessref 使用）

pyhessref 中使用的是"无量纲化"的形式，便于 Krylov 求解：

$$
U_{ai}^A + \frac{A_{ai,bj}}{\varepsilon_a - \varepsilon_i} U_{bj}^A = -\frac{B_{ai}^A}{\varepsilon_a - \varepsilon_i}
$$

这种形式具有 $U + \text{resp}(U) = \text{rhs}$ 的标准 Krylov 结构。

### 2.3 实现细节 (`compute_dimensionless_cphf_rhs`)

```python
b1mo = f1mo - s1mo * eocc       # [natm, 3, nmo, nocc]
rhs = np.zeros([natm, 3, nmo, nocc])
rhs[:, :, nocc:, :] = -b1mo[:, :, nocc:, :] / e_ai_shift
rhs[:, :, :nocc, :] = -0.5 * s1mo[:, :, :nocc, :]
```

注意：
- `e_ai_shift = (evir[:, None] - eocc[None, :]) + level_shift`，加 level shift 改善收敛
- `rhs` 的 `nocc:` 部分（虚-占据块）是标准 CP-HF 右端项
- `rhs` 的 `:nocc` 部分（占据-占据块）固定为 `-0.5 * s1mo`，不参与迭代
- 此处合并 `mo1[occ, occ]` 到迭代变量中，是为了让 rhs 的求值不需要 response 计算（与 PySCF 的 `solve_withs1` 思路一致）

## 3. 响应函数 (`response`)

### 3.1 RHF 响应函数的数学形式

在 RHF 中，response 就是 Fock 算符在 perturbed density 上的作用：

$$
\text{resp}(D^{(1)})_{\mu\nu} = 4 J_{\mu\nu}[D^{(1)}] - K_{\mu\nu}[D^{(1)}] - K_{\nu\mu}[D^{(1)}]
$$

其中 $D^{(1)} = C \cdot U \cdot C_\text{occ}^T$（half-transformed）。

### 3.2 Half-transformation 优化

直接构造 `dm1` 是 `[nao, nao]` 矩阵，但 perturbed coefficients 实际上只需要 `[nao, nocc]`。pyhessref 中实现的是：

```python
ubra = mo_coeff @ mo1                 # [..., nao, nocc]
resp_bra = el_obj.get_response_bra(ubra)   # [..., nao, nocc]
resp = mo_coeff.T @ resp_bra          # [..., nmo, nocc]
```

`get_response_bra` 内部：

```python
resp_bra_j = 4 * einsum("uvP, PQ, klQ, Akj, lj, vi -> Aui", ...)   # J
resp_bra_k0 = einsum("uvP, PQ, klQ, Avj, lj, ki -> Aui", ...)      # K (direct)
resp_bra_k1 = einsum("uvP, PQ, klQ, Akj, vj, li -> Aui", ...)      # K (exchange)
resp_bra = resp_bra_j - resp_bra_k0 - resp_bra_k1
```

数值因子 4 来自 RHF 密度矩阵的 2 倍 × 对称化的 2 倍 = 4。

### 3.3 Dimensionless 响应 (`response_dimless_cphf`)

将 dimensionful response 转化为无量纲形式：

```python
resp = self.response_mo(mo1)
if level_shift != 0.0:
    resp -= mo1 * level_shift
resp[..., nocc:, :] /= e_ai_shift      # 虚-占据块除以能量差
resp[..., :nocc, :] = 0                # 占据-占据块强制置零
```

强制将 `:nocc` 部分置零是因为：我们利用 `mo1[occ, occ]` 来表达 overlap 一阶导数的固定贡献，它不参与 Krylov 迭代。

## 4. Krylov 求解器

### 4.1 Block Krylov 算法

`krylov_block.py` 实现了块 Krylov 子空间方法（求解 $(1+A)x = b$）。它的特点：

- **批处理**：每次迭代同时处理多个右端项（block 形式）
- **非归一化基向量**：基向量保持正交但不归一化，便于通过 $\|v\|^2$ 自然检测收敛
- **PySCF 兼容**：收敛行为与 `pyscf.lib.krylov` 一致

### 4.2 求解流程 (`solve_dimless_cphf`)

```python
rhs_shape = rhs.shape   # [natm, 3, nmo, nocc]
rhs = rhs.reshape(-1, nmo * nocc)   # 展平为 [natm*3, nmo*nocc]

def response_cphf_flattened(x):
    x = x.reshape(-1, nmo, nocc)
    y = self.response_dimless_cphf(x)
    return y.reshape(-1, nmo * nocc)

mo1 = krylov_block(response_cphf_flattened, rhs)
mo1 = mo1.reshape(rhs_shape)
```

## 5. CP-HF 后处理 (`finalize_cphf`)

### 5.1 最后一次迭代修正

Krylov 求解收敛后，需要做最后一次迭代来：
1. 重新计算精确的 `mo1[vir, occ]`（不带 level shift）
2. 得到 `mo_e1`（占据轨道能量的一阶导数 = Fock 矩阵 occ-occ 块的一阶导数）

```python
b1mo = f1mo - s1mo * eocc + self.response_mo(mo1)
mo1[:, :, nocc:, :] = -b1mo[:, :, nocc:, :] / e_ai
mo_e1 = b1mo[:, :, :nocc, :] + mo1[:, :, :nocc, :] * e_ij
```

### 5.2 `mo_e1` 的物理含义

`mo_e1` 是 Fock 矩阵 occ-occ 块的一阶导数，形状 `[natm, 3, nocc, nocc]`：

- 标准 SCF 下，Fock 矩阵在 MO 表象下对角化，所以 `mo_e1` 的对角元就是占据轨道能量的一阶导数
- 由于 $U_{ji}^A = -\frac{1}{2} S_{ji}^{(1),A}$ 的约定，`mo_e1` 的非对角元一般不为零
- 这是 CP-HF 完整解的一个副产品，但 Hessian 构造时需要使用

## 6. CP-HF 对 Hessian 的贡献 (`get_cphf_hess`)

最终的 CP-HF Hessian 贡献由三项组成：

```python
de_cphf[A, B] += 4 * (f1mo[A][:, None] * mo1[B][None, :]).sum(axis=(-1, -2))
de_cphf[A, B] -= 4 * (s1mo[A][:, None] * mo1[B][None, :] * eocc).sum(axis=(-1, -2))
de_cphf[A, B] -= 2 * (s1oo[A][:, None] * mo_e1[B][None, :]).sum(axis=(-1, -2))
```

物理意义：

1. **`4 * f1mo * mo1`**：Fock 一阶导数与轨道响应的耦合（因子 4 = 占据数 2 × 对称性 2）
2. **`-4 * s1mo * mo1 * eocc`**：来自 overlap 一阶导数对应的能量加权部分
3. **`-2 * s1oo * mo_e1`**：来自 Fock 矩阵的非平凡 occ-occ 一阶导数与 overlap 占据-占据导数的耦合（因子 2 = 占据数）

### 6.1 与 PySCF 的对应

PySCF 中 `hessian.rhf.hess_elec` 的等价代码：

```python
de[i0, j0] += 4 * np.einsum('tuv, suv -> ts', h1ao[ia], dm1)
de[i0, j0] -= 4 * np.einsum('tuv, suv -> ts', s1ao, dm1_e)
de[i0, j0] -= 2 * np.einsum('tuv, suv -> ts', s1oo, mo_e1[ja])
```

其中 `dm1 = mo1 @ mocc.T`，`dm1_e = mo1 @ (mocc * occ_energy).T`。pyhessref 中直接在 MO 表象做缩并，避免构造 AO 表象的 dm1。

## 7. 工作流总结

完整的 CP-HF Hessian 计算流程（`make_cphf_hess`）：

```
1. compute_dimensionless_cphf_rhs()
   ↓ 输出 {rhs, f1mo, s1mo}
   
2. make_response_preparation()
   ↓ 为每个 electron interaction 对象准备响应所需的数据

3. solve_dimless_cphf(rhs)
   ↓ Krylov 求解 U + resp(U) = rhs
   ↓ 输出 mo1 (尚未做最后修正)

4. finalize_cphf(mo1, pre_cphf_dict)
   ↓ 输出 {mo1, mo_e1} (修正后)

5. get_cphf_hess(f1mo, s1mo, mo1, mo_e1)
   ↓ 输出 de_cphf [natm, natm, 3, 3]
```
