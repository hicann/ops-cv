# Documentation Center

## Directory Structure

The Docs directory structure is described as follows:

```text
├── zh
│    ├── context                            # Public documents, such as terminology and basic concepts
│    ├── debug                              # Operator debugging guidance documents
│    │   ├── op_debug_prof.md
│    │   ├── ...
│    ├── develop                            # Operator development guidance documents
│    │   ├── aicore_develop_guide.md
│    │   ├── aicpu_develop_guide.md
│    │   ├── ...
│    ├── figures                            # Image directory
│    ├── install                            # Environment installation and compilation guidance documents
│    │   ├── build.md
│    │   ├── compile.md
│    │   ├── quick_install.md
│    │   └── ...
│    ├── invocation                         # Operator invocation guidance documents (including aclnn invocation, graph mode invocation, etc.)
│    │   ├── quick_op_invocation.md
│    │   ├── ...
│    ├── menu_aclnn_api.md                  # Full aclnn interface index file
│    ├── op_api_list.md                     # aclnn interface list
│    ├── op_list.md                         # Full operator list
├── CONTRIBUTING_DOCS.md                 # Documentation contribution instructions
├── QUICKSTART.md                        # Quick start
└── README.md
```

## Advanced Tutorials

### Guide Documents

| Document                                                         | Description                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Source Code Build Guide](zh/install/compile.md)                        | Introduces different source code build methods and verification methods in online and offline scenarios.          |
| [Operator Invocation Guide](zh/invocation/quick_op_invocation.md)         | Introduces operator sample invocation methods and different operator invocation methods (such as PyTorch/aclnn/graph). |
| [Standard Operator Development Guide](zh/develop/aicore_develop_guide.md)       | Introduces how to define operator prototypes and implement Tiling and Kernel based on standard engineering. Such operators are called "standard operators".<br>Standard operators support aclnn and graph mode invocation. |
| [Simple Operator Development Guide](../examples/fast_kernel_launch_example/README.md) | Introduces how to implement fast_kernel_launch based on simple engineering, that is, the `<<<>>>` method. Such operators are called "simple operators" (also known as ecosystem simplest operators).<br>Simple operators only support PyTorch invocation. |
| [Operator Debugging and Tuning](zh/debug/op_debug_prof.md)                    | Introduces common operator function debugging and performance tuning methods (such as data collection and simulation pipeline). |

### API Documents

| Document        | Description                  |
| ----------------------- | ---------------------- |
| [Operator List](zh/op_list.md)                        | Introduces the list of all operators included in the project.                                 |
| [aclnn List](zh/op_api_list.md)                   | Introduces the list of all operator aclnn APIs included in the project. To facilitate users calling operators on the Host side, C language APIs are provided, that is, APIs with the aclnn prefix. |

### Tool Documents

| Document        | Description                  |
| ----------------------- | ---------------------- |
| [Simulator Simulation Tool](zh/debug/cann_sim.md) | SoC-level simulation tool for operator development scenarios, used to analyze accuracy and performance data of AI tasks running on AI simulators at each stage. |

### More Documents

Welcome to visit the [wiki center](https://gitcode.com/cann/ops-math/wiki/Home.md) to learn more project information, including project positioning, operator development supplementary knowledge introduction, performance optimization methodology, frequently asked questions (FAQ), and problem positioning methods.

## Appendix

| Document                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](zh/context/基本概念.md) | Introduces basic concepts and terminology related to the operator domain, such as quantization/sparse, data type, and data format. |
| [Build Parameter Description](zh/install/build.md)   | Introduces the functions and values of build.sh parameters in this project, including source code compilation, operator invocation, and debugging. |
