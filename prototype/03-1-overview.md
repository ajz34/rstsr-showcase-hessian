# RI-RHF Hessian 实现概览

**LLM AI 生成提示**：该文档由 AI 编写，目前暂没有人工校对。如项目推进中遇到该文档存在问题，请联系维护者进行修正。

## 1. Hessian 分解公式

RI-RHF 方法的 Hessian 可以分解为以下贡献项之和：

$$
E_{\text{hess}} = \underbrace{E_{\text{nuc}}}_{\text{核排斥}} + \underbrace{E_{\text{hcore}}}_{\text{核心哈密顿}} + \underbrace{E_{\text{ovlp}}}_{\text{重叠积分}} + \underbrace{E_J^{(20)} + E_J^{(11)} + E_J^{(02)}}_{\text{RI-J skeleton}} - \frac{1}{2}\underbrace{\left(E_K^{(20)} + E_K^{(11)} + E_K^{(02)}\right)}_{\text{RI-K skeleton}} + \underbrace{E_{\text{cphf}}}_{\text{CP-HF 响应}}
$$

其中上标 $(mn)$ 的含义是基组导数阶数为 $m$、辅助基导数阶数为 $n$。K 项前的 $1/2$ 来源于 RHF 中交换积分的系数为 $-1/2$（相对于库伦积分的系数 1）。

### 各项含义简述

| 贡献项 | 密度矩阵依赖 | 计算复杂度 | 说明 |
|--------|------------|-----------|------|
| $E_{\text{nuc}}$ | 无 | $O(n^2)$ | 原子核排斥能二阶导数，纯几何量 |
| $E_{\text{hcore}}$ | 一阶 ($\text{tr}(D \cdot H'')$) | $O(n^2)$ | 核心哈密顿 skeleton 导数 |
| $E_{\text{ovlp}}$ | 一阶 ($\text{tr}(D_e \cdot S'')$) | $O(n^2)$ | 重叠积分导数贡献（来自 Hellmann-Feynman 定理） |
| $E_J^{(20)}$ | 二阶 ($D \otimes D$) | $O(n^4)$ | 基组二阶导数、辅助基零阶导数的 J 贡献 |
| $E_J^{(11)}$ | 二阶 | $O(n^4)$ | 基组一阶、辅助基一阶导数的 J 贡献 |
| $E_J^{(02)}$ | 二阶 | $O(n^4)$ | 基组零阶、辅助基二阶导数的 J 贡献 |
| $E_K^{(mn)}$ | 二阶 | $O(n^4)$ | 交换积分对应贡献，结构类似 J |
| $E_{\text{cphf}}$ | 响应（密度矩阵导数） | 迭代求解 | 来自密度矩阵一阶响应的耦合贡献 |

## 2. 项目文件结构

### prototype/ — 自上而下的原型实现

| 文件 | 内容 |
|------|------|
| `00-ref_moles.ipynb` | 计算参考数据（执行时间长，不应重跑） |
| `nh3_r_hf.npz` | 起始数据：mo_coeff, mo_occ, mo_energy 等 |
| `01-decomp_nh3_r.ipynb` | Hessian 分解概览，利用 PySCF 的 `auxbasis_response` 参数分离 J/K 的 $(20)/(11)/(02)$ 贡献 |
| `02-1-decomp_e0e1.ipynb` | 核排斥、hcore、overlap 的详细分解 |
| `02-2-decomp_de_J.ipynb` | RI-J skeleton 二阶导数的逐项分解 |
| `02-3-decomp_de_K.ipynb` | RI-K skeleton 二阶导数的逐项分解 |
| `02-4-decomp_cphf_1.ipynb` | CP-HF 的 f1ao、s1ao 构造与右端项 |
| `02-5-decomp_cphf_2.ipynb` | CP-HF 方程求解与 response 函数 |
| `02-6-decomp_cphf_3.ipynb` | Krylov 求解器的实现与比较 |
| `nh3_r_hf_decomp.npz` | 所有分解结果与中间变量的存储文件 |
| `krylov_block.py` | 独立的 block Krylov 求解器 |

### pyhessref/ — 自下而上的工程化实现

| 文件 | 内容 |
|------|------|
| `util.py` | 通用工具：dm0/dme0 生成 |
| `nuc_repl.py` | 核排斥 Hessian |
| `hcore.py` | 核心哈密顿 skeleton 导数（一阶、二阶） |
| `ovlp.py` | 重叠积分导数贡献 |
| `hess_trait_restricted.py` | RHF Hessian 的抽象接口定义 |
| `hess_impl_restricted.py` | RHF Hessian 的求解器（CP-HF 流程） |
| `krylov_block.py` | Block Krylov 求解器 |
| `rijk/hess_restricted_naive.py` | RI-JK 的 naive 实现 |
| `tests/test_hessian_rhf_naive.py` | 完整的 pytest 测试 |

## 3. 三层代码的关系

### PySCF 源码 → prototype

这是"反向工程"过程，目的是理解并分解 PySCF 的 Hessian 计算：

1. **参考数据获取**：通过 `mf.Hessian().run()` 获得最终的 `de_ref`，以及通过调节 `auxbasis_response` 参数获得不同辅助基响应阶数下的 partial hess_ejk
2. **分离 $(20)/(11)/(02)$**：利用 PySCF 的 `auxbasis_response` 参数（0/1/2）获得不同阶数贡献的线性组合，再通过差分提取各阶：
   - $J_{20} = e_{j}^{(0)}$
   - $J_{11} = 2(e_j^{(1)} - e_j^{(0)})$
   - $J_{02} = e_j^{(2)} - 2e_j^{(1)} + e_j^{(0)}$
3. **逐项展开**：对每个子项（如 J20_1, J20_2 等），从数学公式出发，用 einsum 直接表达，与 PySCF 的中间结果进行逐项比对验证
4. **关键设计**：prototype 中使用完整积分（不考虑对称性），用 einsum 表达（方便理解与验证），不追求效率

### prototype → pyhessref

这是"工程化重构"过程：

1. **抽象接口设计**：将 Hessian 的各组成部分抽象为 `RHessCoreAPI`（零/一阶密度矩阵项）和 `RHessElecInteractAPI`（二阶及以上项）两个抽象基类
2. **CP-HF 流程重构**：引入"dimensionless" CP-HF 公式，将标准 CP-HF `(ea-ei)U - AU = B` 转化为 `U + resp(U) = rhs`，便于 Krylov 求解
3. **half-transformation**：在 `get_deriv1_bra` 和 `get_response_bra` 中使用半变换（只变换 ket 端到占据轨道），避免构造完整的 nao×nao 矩阵
4. **测试驱动**：每个组件都有对应测试，与 `nh3_r_hf_decomp.npz` 中的参考数据比对

## 4. 通用变量与维度

| 变量名 | 维度 | 含义 |
|--------|------|------|
| `mo_coeff` | `[nao, nmo]` | 分子轨道系数 |
| `mo_occ` | `[nmo]` | 轨道占据数 |
| `mo_energy` | `[nmo]` | 轨道能量 |
| `mocc` | `[nao, nocc]` | 占据轨道系数 |
| `dm0` | `[nao, nao]` | 密度矩阵（RHF: $2 C_\text{occ} C_\text{occ}^T$） |
| `dme0` | `[nao, nao]` | 能量加权密度矩阵（RHF: $2 C_\text{occ} \text{diag}(\varepsilon_\text{occ}) C_\text{occ}^T$） |
| `mocc_2` | `[nao, nocc]` | 带占据数开根的占据轨道：$C_\mu \sqrt{\text{occ}_i}$ |
| `int3c2e` | `[nao, nao, naux]` | 三中心积分 $(\mu\nu|P)$ |
| `int2c2e` | `[naux, naux]` | 两中心积分 $(P|Q)$ |
| `int2c2e_inv` | `[naux, naux]` | 两中心积分的逆矩阵 |
| `f1mo` | `[natm, 3, nmo, nocc]` | Fock 矩阵一阶导数的 MO 表象 |
| `s1mo` | `[natm, 3, nmo, nocc]` | 重叠矩阵一阶导数的 MO 表象 |
| `mo1` | `[natm, 3, nmo, nocc]` | CP-HF 求解的轨道响应系数 $U_{pi}^A$ |
| `mo_e1` | `[natm, 3, nocc, nocc]` | 占据轨道能量的一阶导数 |

## 5. NH3 测试分子

所有原型实现使用 NH3 分子（非对称构型）：

```
N  0   0   0
H  1.0 0.1 0.2
H  0.3 1.1 0.2
H  0.1 0.1 1.2
```

基组：def2-TZVP，密度拟合。4 个原子，每个原子 3 个笛卡尔坐标，Hessian 维度为 `[4, 4, 3, 3]`。
