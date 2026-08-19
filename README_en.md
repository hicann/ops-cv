# ops-cv

English | [简体中文](./README.md)

## 🔥Latest News

- [2026/01] Added [QuickStart](docs/QUICKSTART.md) to guide beginners with zero foundation in operator project deployment (supporting Docker environment), operator development, and contribution process.
- [2025/12] Open-source operators support Ascend 950PR/Ascend 950DT/KirinX90, and can be developed and debugged through the [CANN Simulator](docs/zh/debug/cann_sim.md) simulation tool; optimized guide documents, focused on the [Operator Development Guide](docs/zh/develop/aicore_develop_guide.md), clarified minimal deliverables and key sample code, and provided migration guidance for this project for Ascend/samples repository operators; added ONNX operator plugin support.
- [2025/11] The [three_interpolate_backward](image/three_interpolate_backward/docs/aclnnThreeInterpolateBackward.md) operator added deterministic computation feature; added [opgen](docs/zh/develop/aicore_develop_guide.md#工程创建) support for automatically generating operator projects.
- [2025/10] Added experimental directory, improved [Contribution Guide](CONTRIBUTING_en.md), supporting developers to debug and contribute custom operators.
- [2025/09] The ops-cv project was launched for the first time, and open-source operators support Atlas A2/A3 series products.

## 🚀Overview

ops-cv is a high-level operator library in the [CANN](https://hiascend.com/software/cann) (Compute Architecture for Neural Networks) operator library that provides image processing, object detection, and other capabilities, covering common image processing operations, including image-class and objdetect-class operators. The operator library architecture diagram is as follows:

<img src="docs/zh/figures/architecture.png" alt="Architecture Diagram"  width="700px" height="326px">

## 📝Version Compatibility

The source code of this project will be released with the CANN software version. For the correspondence between the CANN software version and the tags of this project, refer to the corresponding version description in the [release repository](https://gitcode.com/cann/release-management).
Note that to ensure your source code customization development proceeds smoothly, select the compatible CANN version and Gitcode tag source code. Using the master branch may pose a risk of version mismatch.

## ⚡️Quick Start

If you want to **quickly experience the project capabilities from scratch**, visit the following simple tutorials.

1. [Environment Deployment](docs/zh/install/quick_install.md): This article is the **prerequisite for QuickStart and various tutorials**. Please complete the basic environment setup and source code download first.

2. [QuickStart](docs/QUICKSTART.md): Provides a quick-start simple guide for key capabilities such as source code compilation, operator invocation/development/debugging.

## 📖Learning Tutorials

If you have completed the **Quick Start** section, have a certain understanding of this project, and want to **deeply understand and experience the project**, visit the following detailed tutorials.

1. [Operator List](docs/zh/op_list.md): Provides full operator information, making it easy for you to view operator classifications and functions.
2. [Operator Invocation](docs/zh/invocation/quick_op_invocation.md): Provides multiple methods for source code compilation and execution of operator samples (including executing UT).
3. [Operator Development](docs/zh/develop/aicore_develop_guide.md): Provides an end-to-end operator development guide, learning from scratch to create operator projects and implement Tiling and Kernel core deliverables.
4. [Operator Debugging and Tuning](docs/zh/debug/op_debug_prof.md): Provides common operator debugging and tuning methods, such as DumpTensor, Msprof, and Simulator.

In addition to the above key tutorials, there are other document introductions, such as operator invocation methods, build parameter descriptions, and terminology concepts. For full documentation, visit [docs](docs/README.md).

## 🔍Directory Structure

The key directories are as follows. For detailed directory introduction, refer to [Project Directory](./docs/zh/install/dir_structure.md).

```Cpp
├── build.sh                       # Project engineering compilation script
├── cmake                          # Project engineering compilation directory
├── CMakeLists.txt
├── common                         # Project common header files and common source code
├── docs                           # Project documentation introduction (zh for Chinese, en for English)
├── examples                       # End-to-end operator development and invocation examples
├── experimental                   # User-defined operator storage directory
├── objdetect                      # objdetect-class operators
├── ...
├── image                          # image-class operators
│   ├── grid_sample                # GridSample operator all deliverables, such as Kernel, InferShape, etc.
│   │   ├── CMakeLists.txt         # Operator compilation configuration file
│   │   ├── docs                   # Operator description document
│   │   ├── examples               # Operator usage examples
│   │   ├── op_graph               # Operator graph composition related directory
│   │   ├── op_host                # Operator information library, Tiling, InferShape related implementation directory
│   │   ├── op_api                 # Optional, operator aclnn interface implementation directory. If not provided, the aclnn interface of this operator will be automatically generated by the project
│   │   ├── op_kernel              # Operator Kernel directory
│   │   └── README.md              # Operator introduction document
│   ├── ...
│   └── CMakeLists.txt             # Operator compilation configuration file
├── tests                          # Test engineering directory
├── README.md
├── install_deps.sh                # Install dependency package script
├── requirements.txt               # Third-party dependency packages required by the project
└── scripts                        # Script directory, containing custom operator and Kernel build related configuration files
```

## 💬Related Information

- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
- [Affiliated SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

## 🤝Contact Us

The functions and documentation of this project are being continuously updated and improved. We recommend that you follow the latest version.

- **Problem Feedback**: Submit problems through GitCode [Issues](https://gitcode.com/cann/ops-cv/issues).
- **Community Interaction**: Participate in communication through GitCode [Discussions](https://gitcode.com/cann/ops-cv/discussions).
- **Technical Column**: Obtain technical articles through GitCode [Wiki](https://gitcode.com/cann/ops-cv/wiki), such as serialized tutorials and excellent practices.
