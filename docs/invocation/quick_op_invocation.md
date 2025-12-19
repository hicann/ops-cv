# 算子调用
## 前提条件
- 环境部署：调用项目算子之前，请先参考[环境部署](../context/quick_install.md)完成基础环境搭建。
- 调用算子列表：项目可调用的算子参见[算子列表](../op_list.md)，算子对应的aclnn接口参见[aclnn列表](../op_api_list.md)。

## 编译执行

若基于社区版CANN包对算子源码修改，可使用[自定义算子包](#自定义算子包)和[ops-cv包](#ops-cv包)方式编译执行。

- 自定义算子包：选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。注意自定义算子包优先级高于原始CANN包。
- ops-cv包：选择整个项目编译生成的包称为ops-cv包，可**完整替换**CANN包对应部分。

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：
    
    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 以GridSample算子编译为例
    # bash build.sh --pkg --soc=ascend910b --ops=grid_sample
    # 编译experimental目录下的用户算子
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=grid_sample
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子（参见[算子列表](../op_list.md)）。格式形如"grid_sample,iou_v2,..."，多算子之间用英文逗号","分隔。
    - --experimental（可选）：表示编译用户保存在experimental目录下的算子。
    说明：若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是built-in包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。

    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-cv-${vendor_name}_linux-${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。
    
2. **安装自定义算子包**
   
    ```bash
    ./build_out/cann-ops-cv-${vendor_name}_linux-${arch}.run
    ```
    
    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann。

3. **（可选）删除自定义算子包**

    注意自定义算子包不支持卸载，可通过如下操作删除：

    请删除`vendors/${vendor_name}_cv`目录，并删除vendors/config.ini中load\_priority对应\${vendor_name}\_cv的配置项。

### ops-cv包

1. **编译ops-cv包**

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译除experimental目录外的所有算子
    bash build.sh --pkg [--jit] --soc=${soc_version}
    # 编译experimental目录下的用户算子
    # bash build.sh --pkg --experimental [--jit] --soc=${soc_version}
    ```
    - --jit（可选）：设置后表示不编译算子二进制文件，如需使用aclnn调用算子，该选项无需设置。
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --experimental（可选）：表示编译用户保存在experimental目录下的算子。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-cv_${cann_version}_linux-${arch}.run" successfully created.
    ```

   \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-cv包**

    ```bash
    ./build_out/cann-${soc_name}-ops-cv_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **（可选）卸载ops-cv包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/ops_cv/script/uninstall.sh
    ```
## 本地验证 

通过项目根目录build.sh脚本，可快速调用算子和UT用例，验证项目功能是否正常，build参数介绍参见[build参数说明](../context/build.md)。目前算子支持API方式（aclnn接口）和图模式调用，**推荐aclnn调用**。

- **执行算子样例**
  
    - 完成ops-cv包安装后，执行命令如下：
        ```bash
        bash build.sh --run_example ${op} ${mode}
        # 以GridSample算子example执行为例
        # bash build.sh --run_example grid_sample eager
        ```
        
        - \$\{op\}：表示待执行算子，算子名小写下划线形式，如grid_sample。
        - \$\{mode\}：表示算子执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
    
    - 完成自定义算子包安装后，执行命令如下：
        ```bash
        bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}]
        # 以GridSample算子example执行为例
        # bash build.sh --run_example grid_sample eager cust --vendor_name=custom
        ```
    
        - \$\{op\}：表示待执行算子，算子名小写下划线形式，如grid_sample。
        - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
        - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。         
        - \$\{vendor\_name\}（可选）：与构建的自定义算子包设置一致，默认名为custom。        
        
        说明：\$\{mode\}为graph时，不指定\$\{pkg_mode\}和\$\{vendor\_name\}

    执行算子样例后会打印执行结果，以GridSample算子为例，结果如下：

    ```
    This environment does not have the ASAN library, no need enable ASAN
    CMAKE_ARGS: -DENABLE_UT_EXEC=TRUE
    ----------------------------------------------------------------
    Start to run examples,name:grid_sample mode:eager
    Start compile and run examples file: ../image/grid_sample/examples/test_aclnn_grid_sample2_d.cpp
    pkg_mode:cust vendor_name:custom
    resultData[0] is: 0.250000
    resultData[1] is: 2.250000
    resultData[2] is: 2.000000
    resultData[3] is: 8.500000
    resultData[4] is: 20.500000
    resultData[5] is: 12.000000
    resultData[6] is: 8.250000
    resultData[7] is: 18.250000
    resultData[8] is: 10.000000
    Start compile and run examples file: 
    ../image/grid_sample/examples/test_aclnn_grid_sample3_d.cpp
    pkg_mode:cust vendor_name:custom
    resultData[0] is: 0.250000
    resultData[1] is: 0.875000
    resultData[2] is: 2.000000
    resultData[3] is: 4.000000
    ```
- **执行算子UT**

	> 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

    ```bash
  # 安装根目录下test相关requirements.txt依赖
  pip3 install -r tests/requirements.txt
  # 方式1: 编译并执行指定算子和对应功能的UT测试用例（选其一）
  bash build.sh -u --[opapi|ophost|opkernel] --ops=iou_v2
  # 方式2: 编译并执行所有的UT测试用例
  # bash build.sh -u
  # 方式3: 编译所有的UT测试用例但不执行
  # bash build.sh -u --noexec
  # 方式4: 编译并执行对应功能的UT测试用例（选其一）
  # bash build.sh -u --[opapi|ophost|opkernel]
  # 方式5: 编译对应功能的UT测试用例但不执行（选其一）
  # bash build.sh -u --noexec --[opapi|ophost|opkernel]
    ```

    假设验证ophost功能是否正常，执行如下命令：
    ```bash
  bash build.sh -u --ophost
    ```

    执行完成后出现如下内容，表示执行成功。
    ```bash
  Global Environment TearDown
  [==========] ${n} tests from ${m} test suites ran. (${x} ms total)
  [  PASSED  ] ${n} tests.
  [100%] Built target cv_op_host_ut
    ```
    \$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。
