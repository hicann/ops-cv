# ops-cv

## 🔥Latest News

- [2026/01] 新增[QuickStart](docs/QUICKSTART.md)，指导新手零基础入门算子项目部署（支持Docker环境）、算子开发和贡献流程。
- [2025/12] 开源算子支持Ascend 950PR/Ascend 950DT/KirinX90，可以通过[CANN Simulator](docs/zh/debug/cann_sim.md)仿真工具开发调试；优化指南类文档，聚焦[算子开发指南](docs/zh/develop/aicore_develop_guide.md)，明确最小交付件和关键示例代码，针对Ascend/samples仓算子提供迁移本项目的指导；新增onnx算子插件支持。
- [2025/11] [three_interpolate_backward](image/three_interpolate_backward/docs/aclnnThreeInterpolateBackward.md)算子新增确定性计算特性；新增[opgen](docs/zh/develop/aicore_develop_guide.md#工程创建)支持自动生成算子工程。
- [2025/10] 新增experimental目录，完善[贡献指南](CONTRIBUTING.md)，支持开发者调试并贡献自定义算子。
- [2025/09] ops-cv项目首次上线，开源算子支持Atlas A2/A3系列产品。

## 🚀概述

ops-cv是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供图像处理、目标检测等能力的高阶算子库，涵盖常见的图像处理操作，包括image类、objdetect类算子，算子库架构图如下：

<img src="docs/zh/figures/architecture.png" alt="架构图"  width="700px" height="326px">

## 📝版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。
请注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与Gitcode标签源码，使用master分支可能存在版本不匹配的风险。

## ⚡️快速入门

若您希望**从零到一快速体验**项目能力，请访问下述简易教程。

1. [环境部署](docs/zh/install/quick_install.md)：本文是**QuickStart和各类教程的操作前提**，请先完成基础环境搭建和源码下载。

2. [QuickStart](docs/QUICKSTART.md)：针对源码编译、算子调用/开发/调试等关键能力，提供快速上手的简易指南。

## 📖学习教程

若您已学习完**快速入门**章节，对本项目有一定认知，并希望**深入了解和体验项目**，请访问下述详细教程。

1. [算子列表](docs/zh/op_list.md)：提供全量算子信息，方便您查看算子分类和功能。
2. [算子调用](docs/zh/invocation/quick_op_invocation.md)：提供多种源码编译和执行算子样例（包括执行UT）的方法。
3. [算子开发](docs/zh/develop/aicore_develop_guide.md)：提供算子端到端开发指南，从零学习创建算子工程、实现Tiling和Kernel核心交付件。
4. [算子调试调优](docs/zh/debug/op_debug_prof.md)：提供常见算子调试和调优方法，如DumpTensor、Msprof、Simulator等。

除了上述关键教程，还有其他文档介绍，例如算子调用方式、build参数说明、术语概念等，全量文档请访问[docs](docs/README.md)。

## 🔍目录结构

关键目录如下，详细目录介绍参见[项目目录](./docs/zh/install/dir_structure.md)。

```Cpp
├── build.sh                       # 项目工程编译脚本
├── cmake                          # 项目工程编译目录
├── CMakeLists.txt
├── common                         # 项目公共头文件和公共源码                 
├── docs                           # 项目文档介绍(zh为中文，en为英文) 
├── examples                       # 端到端算子开发和调用示例
├── experimental                   # 用户自定义算子存放目录
├── objdetect                      # objdetect类算子
├── ...
├── image                          # image类算子
│   ├── grid_sample                # GridSample算子所有交付件，如Kernel、InferShape等
│   │   ├── CMakeLists.txt         # 算子编译配置文件
│   │   ├── docs                   # 算子说明文档
│   │   ├── examples               # 算子使用示例
│   │   ├── op_graph               # 算子构图相关目录
│   │   ├── op_host                # 算子信息库、Tiling、InferShape相关实现目录
│   │   ├── op_api                 # 可选，算子aclnn接口实现目录，如未提供则表示此算子的aclnn接口会让工程自动生成
│   │   ├── op_kernel              # 算子Kernel目录
│   │   └── README.md              # 算子介绍文档
│   ├── ...
│   └── CMakeLists.txt             # 算子编译配置文件
├── tests                          # 测试工程目录
├── README.md
├── install_deps.sh                # 安装依赖包脚本
├── requirements.txt               # 项目需要的第三方依赖包
└── scripts                        # 脚本目录，包含自定义算子、Kernel构建相关配置文件
```

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-cv/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-cv/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-cv/wiki)获取技术文章，如系列化教程、优秀实践等。
