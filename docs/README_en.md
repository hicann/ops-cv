# Project Documentation

## Directory Description

The key directory structure is as follows:

```Cpp
├── context                            # Public documents, such as terminology and basic concepts
├── debug                              # Operator debugging guidance documents
│   ├── op_debug_prof.md
│   ├── ...
├── develop                            # Operator development guidance documents
│   ├── aicore_develop_guide.md
│   ├── aicpu_develop_guide.md
│   ├── ...
├── figures                            # Image directory
├── install                            # Environment installation guidance documents
│   ├── quick_install.md   
│   ├── build.md               
│   └── ...
├── invocation                         # Operator invocation guidance documents (including aclnn invocation, graph mode invocation, etc.)
│   ├──op_invocation.md
│   ├── ...
├── op_api_list.md                     # Full operator interface list (aclnn)
├── op_list.md                         # Full operator list         
└── README
```

## Document Description

The full project documents are as follows. Please obtain the corresponding content as needed.

| Document                                             | Description                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| [Operator List](zh/op_list.md)                        | Introduces the list of all operators included in the project.                                 |
| [aclnn List](zh/op_api_list.md)                   | Introduces all operator APIs included in the project. Operators can be directly invoked through this API.             |
| [Environment Deployment](zh/install/quick_install.md)          | Introduces the basic environment setup process, including software package and third-party dependency acquisition and installation in different scenarios. |
| [Operator Invocation](zh/invocation/quick_op_invocation.md) | Introduces how to compile source code and execute operators, including operator package compilation, operator sample execution, and UT execution in different scenarios. |
| [Operator Development](zh/develop/aicore_develop_guide.md)   | Introduces how to develop new operators based on this project engineering, including operator prototype definition, Tiling implementation, and Kernel implementation. |
| [Operator Invocation Methods](zh/invocation/op_invocation.md)   | Introduces multiple operator invocation methods and invocation processes, such as aclnn invocation and graph mode invocation. |
| [Operator Debugging and Tuning](zh/debug/op_debug_prof.md)        | Introduces common operator debugging and tuning methods.                               |

## Appendix

| Document                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](zh/context/基本概念.md) | Introduces basic concepts and terminology related to the operator domain, such as quantization/sparse, data type, and data format. |
| [Build Parameter Description](zh/install/build.md)   | Introduces the functions and parameter meanings of the build.sh script in this project.                 |
