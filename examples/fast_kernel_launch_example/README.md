# AscendOps

**AscendOps** - 一个轻量级，高性能的算子开发工程模板

## 项目简介
AscendOps 是一个轻量级，高性能的算子开发工程模板，它集成了PyTorch、PyBind11和昇腾CANN工具链，提供了从算子内核编写，编译到Python封装的完整工具链。

## 核心特性
🚀 开箱即用 (Out-of-the-Box): 预置完整的昇腾NPU算子开发环境配置，克隆后即可开始开发。

🧩 极简设计 (Minimalist Design): 代码结构清晰直观，专注于核心算子开发流程。

⚡ 高性能 (High Performance): 基于AscendC编程模型，充分发挥昇腾NPU硬件能力。

📦 一键部署 (One-Click Deployment): 集成setuptools构建系统，支持一键编译和安装。

🔌 PyTorch集成 (PyTorch Integration): 无缝集成PyTorch张量操作，支持自动微分和GPU/NPU统一接口。

## 核心交付件
1. `csrc/xxx/xxx_torch.cpp` 算子Kernel实现
2. `csrc/xxx/CMakeLists.txt` 算子cmake配置
3. `csrc/npu_ops_def.cpp` 注册算子接口

## 环境要求

1. 参考[前提条件](../../docs/zh/invocation/quick_op_invocation.md#前提条件)的“安装依赖”和以下依赖包清单，完成依赖安装。其中，Python版本要求大于等于3.8。
   - PyTorch: 2.1.0+
   - Ascend Extension for PyTorch

2. 请参考[前提条件](../../docs/zh/invocation/quick_op_invocation.md#前提条件)完成驱动与固件的安装。

## 环境准备

1. **安装社区版CANN toolkit包**

    开发算子前，请参考[环境准备](../../docs/zh/invocation/quick_op_invocation.md#环境准备)完成环境搭建。

2. **配置环境变量**

   根据实际场景，选择合适的命令。

    ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 指定路径安装
   # source ${install_path}/ascend-toolkit/set_env.sh
    ```
3. **安装torch与torch_npu包**

   包含以下两种安装方式，任选其一安装即可：

   - （方式一）下载软件包进行安装：
     1. 根据实际环境，下载对应torch包并安装: `torch-${torch_version}+cpu-${python_version}-linux_${arch}.whl` 下载链接为:[官网地址](http://download.pytorch.org/whl/torch)

        安装命令如下：

        ```sh
        pip install torch-${torch_version}+cpu-${python_version}-linux_${arch}.whl
        ```

     2. 根据实际环境，安装对应torch-npu包: `torch_npu-${torch_version}-${python_version}-linux_${arch}.whl`

        - \$\{torch\_version\}：表示torch包版本号。
        - \$\{python\_version\}：表示python版本号。
        - \$\{arch\}：表示CPU架构，如aarch64、x86_64。

   - （方式二）使用pip命令下载安装:

     ```sh
     pip install torch
     pip install torch_npu
     ```

## 安装步骤

1. 下载源码，进入目录，安装依赖。
    ```sh
    git clone https://gitcode.com/cann/ops-cv-dev.git
    cd ops-cv-dev/examples/fast_kernel_launch_example
    pip install -r requirements.txt
    ```

2. 从源码构建.whl包。
    ```sh
    python -m build --wheel -n
    ```

3. 进入到dist目录，安装构建好的.whl包。
   - 首次安装使用以下命令：
     ```sh
     cd dist
     pip install *.whl
     ```
     打印`Successfully installed ascend-ops-0.0.1`即为安装成功。

   - 重新安装请使用以下命令覆盖已安装过的版本：
     ```sh
     pip install dist/*.whl --force-reinstall --no-deps
     ```
     打印`Successfully installed ascend-ops-0.0.1`即为安装成功。

4. （可选）再次构建前建议先执行以下命令清理编译缓存。
   ```sh
    python setup.py clean
    ```

## 使用示例

安装完成后，您可以像使用普通PyTorch操作一样使用NPU算子，以upsample_nearest3d算子为例，您可以在`ops-cv-dev/examples/fast_kernel_launch_example/ascend_ops/csrc/upsample_nearest3d/test`目录下找到并执行脚本`test_upsamplenearest3d.py`:
```sh
python test_upsamplenearest3d.py
```

```python
import torch
import torch_npu
import ascend_ops

supported_dtypes = {torch.float16, torch.bfloat16, torch.float}
for data_type in supported_dtypes:
    print(f"DataType = <{data_type}>")
    x = torch.randn(1, 3, 8, 20, 45).to(data_type)
    print(f"Tensor x = {x}")
    size = (16, 40, 90)
    if data_type == torch.float :
        cpu_result = torch.nn.functional.interpolate(x, size=size, mode='nearest')
    else :
        cpu_result = torch.nn.functional.interpolate(x.float(), size=size, mode='nearest').to(data_type)
    print(f"cpu: upsample_nearest3d(x, size) = {cpu_result}")
    x_npu = x.npu()
    npu_result = torch.ops.ascend_ops.upsample_nearest3d(x_npu, size).cpu()
    print(f"[OK] torch.ops.ascend_ops.upsample_nearest3d<{data_type}> successfully!")
    print(f"npu: upsample_nearest3d(x, size) = {npu_result}")
    print(f"compare CPU Result vs NPU Result: {torch.allclose(cpu_result, npu_result)}\n\n")
```

最终看到如下输出，即为执行成功：
```bash
compare CPU Result vs NPU Result: True
```

## 开发新算子

   1. 新建算子目录，例如`mykernel`
      
      ```c++
      cd ops-cv-dev/examples/fast_kernel_launch_example/ascend_ops/csrc/
      mkdir mykernel
      cd mykernel
      ```

   2. 编写算子调用文件，例如`mykernel_torch.cpp`。可参考[示例算子](./ascend_ops/csrc/upsample_nearest3d/upsample_nearest3d_torch.cpp)的实现内容。

3. 在`mykernel`目录下创建`CMakeLists.txt`。

    将如下样例中的mykernel，替换为自己的算子名称。
    ```cmake
    message(STATUS "BUILD_TORCH_OPS ON in mykernel")
    # MYKERNEL operation sources
    file(GLOB MYKERNEL_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

    set(MYKERNEL_SOURCES ${MYKERNEL_NPU_SOURCES})
    # Mark .cpp files with special properties
    set_source_files_properties(
        ${MYKERNEL_NPU_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "--npu-arch=dav-2201 -xasc"
    )

    # Create object library
    add_library(mykernel_objects OBJECT ${MYKERNEL_SOURCES})

    target_compile_options(mykernel_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_include_directories(mykernel_objects PRIVATE ${COMMON_INCLUDE_DIRS})
    return()
    ```

4. 在 `ascend_ops/csrc/npu_ops_def.cpp`中添加TORCH_LIBRARY定义。

    ```c++
    TORCH_LIBRARY(ascend_ops, m) {
        m.def("mykernel(Tensor x) -> Tensor");
    }
    ```

5. （可选）在 `ascend_ops/ops.py`中封装自定义接口。
    ```python
    def mykernel(x: Tensor) -> Tensor:
        return torch.ops.ascend_ops.mykernel.default(x)
    ```

6. 参考[安装步骤](#安装步骤)中的步骤2和步骤3进行构建和安装。

7. 编写测试脚本并测试新算子。
    ```python
    torch.ops.ascend_ops.mykernel(x)
    ```
