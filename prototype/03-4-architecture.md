# 代码架构与开发流程

**LLM AI 生成提示**：该文档由 AI 编写，目前暂没有人工校对。如项目推进中遇到该文档存在问题，请联系维护者进行修正。

## 1. pyhessref 代码架构设计

### 1.1 抽象接口层级

pyhessref 采用两层抽象设计，将 Hessian 的不同贡献项按"依赖密度矩阵的阶数"分类：

```
RHessCoreAPI (ABC)               — 零/一阶贡献
├── HessNucRepl                  — 核排斥（零阶，不依赖密度矩阵）
└── RHessHcore                   — 核心哈密顿（一阶，线性于密度矩阵）

RHessElecInteractAPI (ABC)       — 二阶及以上贡献（电子相互作用）
└── RHessRIJKNaive               — RI-JK（二阶，RHF 核心）
    └── get_response_bra(bra)    — 响应函数入口

RHessOvlp (独立类)               — 重叠矩阵导数贡献（特殊处理）
```

**接口分离的原因**：

- **核心项 vs 电子相互作用项**的分离：核心项（hcore、核排斥）只需在 skeleton Hessian 和 CP-HF 右端项（generator_deriv1）中贡献；电子相互作用项（J、K、DFT）还需要在 CP-HF 迭代的 response 中贡献
- **重叠矩阵导数**是独立的：它来自 Hellmann-Feynman 定理的转换，不是 skeleton 导数，且不参与 CP-HF 右端项构造（虽然它提供 `generator_deriv1` 给 s1mo）

### 1.2 RHessCoreAPI 接口

```python
class RHessCoreAPI(ABC):
    @abstractmethod
    def make_skeleton_hess(self, mo_coeff, mo_occ, dm0=None) -> np.ndarray:
        """返回 [natm, natm, 3, 3] 的 skeleton Hessian"""
    
    @abstractmethod
    def generator_deriv1(self) -> callable:
        """返回函数 get_deriv1(A: int) -> [3, nao, nao]"""
```

核心点是：
- `generator_deriv1` 返回一个闭包（closure），避免在 pyhessref 的原子循环中重复打开/关闭各种 context manager
- `make_skeleton_hess` 可以返回零矩阵（如 `HessNucRepl` 的核排斥不依赖密度矩阵，它自己已在前一步做完）

### 1.3 RHessElecInteractAPI 接口

```python
class RHessElecInteractAPI(ABC):
    @abstractmethod
    def make_skeleton_hess(self, mo_coeff, mo_occ) -> np.ndarray:
        """skeleton 二阶导数 Hessian [natm, natm, 3, 3]"""
    
    @abstractmethod
    def get_deriv1_ao(self, mo_coeff, mo_occ) -> np.ndarray:
        """一阶导数在 AO 表象 [natm, 3, nao, nao]"""
    
    def get_deriv1_bra(self, mo_coeff, mo_occ) -> np.ndarray:
        """一阶导数在 half-transformed MO 表象 [natm, 3, nao, nocc]
        默认实现: deriv_ao @ mocc"""
    
    @abstractmethod
    def make_response_preparation(self, mo_coeff, mo_occ):
        """准备响应计算所需数据（存储内部使用）"""
    
    @abstractmethod
    def get_response_bra(self, bra) -> np.ndarray:
        """输入 [..., nao, nocc]，输出 [..., nao, nocc]"""
```

**`get_deriv1_ao` vs `get_deriv1_bra`**：默认 `get_deriv1_bra` 通过 `deriv_ao @ mocc` 实现。但对于 RI-JK，直接在 bra 中利用半变换可以减少一次 `[nao, nocc]` × `[nocc, nocc]` 的矩阵乘法（直接从 `mo1_bra` 计算，避免构造完整的 `foao @ mocc`）。

**`make_response_preparation` 的作用**：对于 RI-JK（naive 版本），response 只需要三中心积分 + 两中心积分逆，不需要预处理。但对于 DFT，可能需要预计算 XC kernel 的某些量。统一的接口允许以相同方式调用。

### 1.4 RHessSCF 主控类

`RHessSCF` 是主控类，组合所有组件：

```python
RHessSCF(
    mol, mo_coeff, mo_occ, mo_energy,
    ovlp_obj=RHessOvlp(mol),
    core_list=[HessNucRepl(mol), RHessHcore(mol)],
    el_list=[RHessRIJKNaive(mol, aux)],
)
```

提供的方法链：
1. `compute_dimensionless_cphf_rhs()` → `{rhs, f1mo, s1mo}`
2. `make_response_preparation()` → 准备 el_list 响应
3. `solve_dimless_cphf(rhs)` → `mo1`
4. `finalize_cphf(mo1, pre_cphf_dict)` → `{mo1, mo_e1}`
5. `get_cphf_hess(f1mo, s1mo, mo1, mo_e1)` → `de_cphf`
6. `make_skeleton_hess()` → 所有 core + el 的 skeleton 和
7. `make_hess()` → 上述所有之和

### 1.5 RHessRIJKNaive 类

```python
class RHessRIJKNaive(RHessElecInteractAPI):
    def __init__(self, mol, aux, scale_j=1.0, scale_k=0.5):
```

- `scale_j` 和 `scale_k` 参数允许调整 J 和 K 的比例系数（RHF 中 J=1, K=1/2；但不同方法的组合如 range-separated 可能需要不同系数）
- `make_skeleton_hess` 内部调用 `get_decomposed_rij_skeleton_deriv2_naive` 和 `get_decomposed_rik_skeleton_deriv2_naive`，结果存储在 `self.result` 中供调试访问
- `get_deriv1_ao` 内部调用 `get_rij_deriv1_ao_naive` 和 `get_rik_deriv1_ao_naive`
- response 部分由 `get_rijk_response_bra_naive` 处理

#### 关于"naive"的含义

"Naive" 在这个上下文中表示：

1. **不利用任何对称性**：所有积分都直接计算完整 `[nao, nao, naux]` 张量
2. **不合并重复计算**：许多中间变量（如 `scr1`、`scr2`、`dbas_*` 等）是分开计算的，虽然它们可能有相同的子表达式
3. **einsum 为主**：所有缩并使用 einsum，便于公式验证和调试
4. **存储所有辅助基导数**：不区分哪些辅助基原子对 Hessian 有实际贡献

## 2. 测试策略

### 2.1 测试架构

测试文件 `tests/test_hessian_rhf_naive.py` 采用基于参考数据的测试方法：

1. **`setUpModule`**：类级别初始化，构造分子、执行 SCF、加载参考数据
2. **数值检查**：每个组件用 `lib.fp()` 生成一个标量指纹，锁定基准值（记录原始浮点数，不依赖外部数据）
3. **对比检查**：与 `nh3_r_hf_decomp.npz` 中的参考数据进行 `np.allclose` 比较

### 2.2 测试项与覆盖

| 测试函数 | 覆盖内容 | 验证对象 |
|----------|---------|---------|
| `test_hess_nuc_repl` | 核排斥 | PySCF 参考值 + np.allclose |
| `test_generator_hcore_deriv2` | hcore generator | PySCF hcore_generator + lib.fp |
| `test_hess_hcore` | hcore skeleton Hessian | 参考数据 + lib.fp |
| `test_hess_ovlp` | overlap 导数 | 参考数据 + lib.fp |
| `test_hess_JK_skeleton_naive` | J/K 的所有子项 | 参考数据 + np.allclose |
| `test_generator_hcore_deriv1` | hcore 一阶导数 | PySCF nuc_grad_method + lib.fp |
| `test_rij_deriv1` | J 一阶导数 | PySCF _gen_jk + lib.fp |
| `test_kij_deriv1` | K 一阶导数 | PySCF _gen_jk + lib.fp |
| `test_f1ao` | f1ao 总组装 | PySCF make_h1 + lib.fp |
| `test_resp_bra` | 响应函数 | PySCF gen_vind + lib.fp |
| `test_dimensionless_cphf_rhs` | 完整 CP-HF 流程 | 参考数据 + np.allclose |
| `test_make_hess` | 总组装 | 参考数据 + lib.fp |

### 2.3 开发中的验证策略

在 prototype → pyhessref 开发过程中，使用的验证策略包括：

1. **中间结果断言**：在每个关键步骤插入 `np.allclose` 断言
2. **参考数据锁定**：将正确结果存储到 `nh3_r_hf_decomp.npz`，每次重构后检查
3. **指纹值锁定**：对每个子贡献项计算 `lib.fp()`，锁定到具体数值（防止参考数据被意外覆盖导致测试退步）
4. **逐步增量测试**：先实现 skeleton 部分，测试通过后再实现 response 部分

## 3. 开发流程：PySCF → prototype → pyhessref

### 3.1 从 PySCF 到 prototype

**目标**：理解 PySCF 源码的 Hessian 计算逻辑，提取出清晰的公式分解

**步骤**：

1. **获取参考数据**：
   - 运行 `mf.Hessian().run()` 得到最终结果
   - 利用 `auxbasis_response=0/1/2` 分离 J/K 的 $(20)/(11)/(02)$ 贡献（具体分离公式见 `01-decomp_nh3_r.ipynb`）
   
2. **理解和利用 PySCF 源码**：
   - PySCF 源码 `pyscf/df/hessian/rhf.py` 和 `pyscf/hessian/rhf.py` 是主要参考
   - PySCF 做了很多优化（如合并重复的积分计算、利用对称性合并项），理解这些优化有助于知道"哪些项应该被合并，哪些应该分离"
   - PySCF 中的 `_gen_jk` 函数是获取 J/K 一阶导数的最直接入口
   
3. **在 prototype 中"从零"实现**：
   - 用 einsum 直接表达公式，忽略效率
   - 与 PySCF 的 `_partial_hess_ejk`、`make_h1`、`gen_vind` 等函数的输出逐项对比
   - 将对比通过的中间结果存入 `nh3_r_hf_decomp.npz`
   
4. **关键挑战与解决**：
   - **挑战**：PySCF 的 `auxbasis_response` 机制需要理解辅助基响应阶数对结果的影响（J 和 K 的缩放系数不同）
   - **解决**：通过线性组合从 `auxbasis_response=0/1/2` 提取 $(20)/(11)/(02)$ 各阶，然后逐一验证
   - **验证**：`de_ref == sum(de_nuc + de_1 + de_J - 0.5*de_K + de_cphf)`

### 3.2 从 prototype 到 pyhessref

**目标**：将 prototype 中的公式实现为工程化、可测试、可扩展的代码

**步骤**：

1. **设计抽象接口**：
   - 分析 prototype 中的计算流程，识别共性模式
   - 设计 `RHessCoreAPI` 和 `RHessElecInteractAPI` 两个抽象层次
   
2. **实现各组件**：
   - 将闭包式的 `generator_*` 函数转换为类方法
   - 将 einsum 实现保留，但加入必要的注释说明公式来源
   
3. **实现 CP-HF 求解器**：
   - 采用 dimensionless 形式，统一 dimesionful/dimensionless 响应
   - 使用 block Krylov 求解器
   
4. **添加完整测试**：
   - 每个组件都有对应测试函数
   - 同时测试功能正确性（与 PySCF 对比）和数值稳定性（锁定 lib.fp 值）

### 3.3 扩展到 UHF 和 DFT

为扩展到 UHF 和 DFT，需要注意：

**UHF 扩展**：
- 抽象接口可以复用（`RHessCoreAPI` 变为 `UHessCoreAPI`）
- core 部分（hcore、nuc）基本相同
- RI-JK 需要分别处理 alpha 和 beta 自旋，密度矩阵变为 $\alpha$ 和 $\beta$ 两部分
- CP-HF 方程变为 2×2 block 结构（alpha 和 beta 耦合）

**DFT 扩展**：
- XC contribution 需要实现 `RHessElecInteractAPI` 接口
- DFT 的 response 与 Fock 不同（需要 XC kernel），所以 `get_response_bra` 需要特殊处理
- RI-JK 部分可以重用
- 总能量表达式中 J 的系数仍然是 1，K 的系数取决于使用的 functional

**通用模式**：
- 对新类型实现 `RHessElecInteractAPI` 接口即可被 `RHessSCF` 的框架调用
- skeleton 部分、response 部分、deriv1 部分彼此独立
- 测试策略相同：先用 prototype 分解验证公式，再在 pyhessref 中工程化

## 4. 代码规范与约定

### 4.1 命名约定

| 类别 | 约定 | 示例 |
|------|------|------|
| 函数 | snake_case | `get_hess_hcore`, `generator_hcore_deriv2` |
| 类 | PascalCase | `RHessHcore`, `RHessRIJKNaive` |
| 抽象类 | CamelCase + API 后缀 | `RHessCoreAPI` |
| 模块 | snake_case | `hess_impl_restricted`, `krylov_block` |
| einsum 中的下标 | 按走向 | `t, s` 为导数方向，`u, v, k, l` 为 AO |

### 4.2 贡献项命名

- `de_J20_1` 表示 J 的 $E^{(20)}$ 部分的第 1 个子贡献项
- `de_K11_4` 表示 K 的 $E^{(11)}$ 部分的第 4 个子贡献项
- `dbas_*` 前缀的变量表示在切片之前、与原子无关的基础贡献矩阵

### 4.3 测试约定

- 测试函数以 `test_` 开头
- 使用 `setUpModule` 做模块级初始化（避免在每个测试函数中重新构造分子）
- 使用 `lib.fp()` 做数值指纹检查（比固定一个绝对值的 assertLess/assertGreater 更稳定）
- 使用 `np.allclose(val, ref, atol=1e-6, rtol=1e-4)` 做对比检查
- 对 iterable 的 ref 值使用字典循环：`for key, val in de_skeleton.items()` 避免遗漏

### 4.4 工程实践

- 所有的积分计算使用 PySCF 的 `mol.intor()` 和 `_int3c_wrapper`，不直接操作底层 C 接口
- 使用闭包/生成器模式来避免重复的 intor 调用（如 `generator_hcore_deriv2`）
- `einsum` 使用 `partial(np.einsum, optimize=True)` 确保在大型张量缩并时获得性能保证
- 使用 `tmp_` 前缀命名临时文件，避免被 git tracked