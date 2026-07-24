# Project Directory

> Some directories listed in this chapter are optional. Please refer to actual deliverables. Especially **single operator directory**, deliverables vary in different scenarios. Specific descriptions are as follows:
>
> - If op_host directory is missing, it may be calling other operator's op_host implementation. See the operator's op_api or op_graph directory source code for calling logic; or Kernel may not have Ascend C implementation yet. If needed, developers are welcome to refer to [Contribution Guide](../../../CONTRIBUTING.md) to supplement and contribute this operator.
> - If op_kernel directory is missing, it may be calling other operator's op_kernel implementation. See the operator's op_api or op_graph directory source code for calling logic; or Kernel may not have Ascend C implementation yet. If needed, developers are welcome to refer to [Contribution Guide](../../../CONTRIBUTING.md) to supplement and contribute this operator.
> - If op_api directory is missing, it means this operator does not support aclnn invocation temporarily.
> - If op_graph directory is missing, it means this operator does not support graph mode invocation temporarily.

Project full directory hierarchy introduction is as follows:

```Cpp
├── cmake                                               # Project engineering compilation directory
│   ├── aclnn_ops_cv.h.in                               # aclnn summary header file template
│   └── ...
├── common                                              # Project common header files and common code
│   ├── CMakeLists.txt
│   ├── inc                                             # Common header file directory
│   └── src                                             # Common code directory
├── experimental                                        # User-defined operator storage directory
│   ├── image                                           # Optional, user-developed image class operator directory
│   │   └── CMakeLists.txt
│   └── objdetect                                       # Optional, user-developed objdetect class operator directory
│       └── CMakeLists.txt
├── ${op_class}                                         # Operator classification, such as image, objdetect class operators
│   ├${op_name}                                         # Operator engineering directory, ${op_name} represents operator name (lowercase underscore form)
│   │   ├── CMakeLists.txt                              # Operator cmakelist entry
│   │   ├── README.md                                   # Operator introduction document
│   │   ├── docs                                        # Operator document directory
│   │   │   └── aclnn${OpName}.md                       # Operator aclnn interface introduction document, ${OpName} represents operator name (PascalCase form)
│   │   ├── examples                                    # Operator invocation example directory
│   │   │   ├── test_aclnn_${op_name}.cpp               # Operator invocation example through aclnn
│   │   │   └── test_geir_${op_name}.cpp                # Operator invocation example through geir
│   │   ├── op_graph                                    # Graph fusion related implementation
│   │   │   ├── CMakeLists.txt                          # op_graph side cmakelist file
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType file, implements operator data type derivation
│   │   │   ├── ${op_name}_proto.h                      # Operator prototype definition, used for operator recognition during graph optimization and fusion phases
│   │   │   └── fusion_pass                             # Operator fusion rule directory
│   │   ├── op_host                                     # Host-side implementation
│   │   │   ├── CMakeLists.txt                          # Host-side cmakelist file
│   │   │   ├── config                                  # Optional, binary configuration file, if not configured project will automatically generate
│   │   │   │   ├── ${soc_version}                      # Operator binary information configured on NPU, ${soc_version} represents NPU model
│   │   │   │   │   ├── ${op_name}_binary.json          # Operator binary configuration file
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # Operator SimplifiedKey configuration information
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # Operator information library, defines operator basic information, such as name, input/output, data types, etc.
│   │   │   ├── ${op_name}_infershape.cpp               # Optional, InferShape implementation, derives output shape according to operator shape, if not configured output shape is same as input shape
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # Optional, Tiling optimization for certain sub-scenarios, ${sub_case} represents sub-scenario, such as ${op_name}_tiling_arch35 is optimization for arch35 architecture, if no such file indicates this operator has no specific Tiling strategy for corresponding sub-scenario
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # Optional, header file used for Tiling implementation in ${sub_case} sub-scenario
│   │   │   ├── ${op_name}_tiling.cpp                   # Optional, if no such file indicates no Tiling implementation for corresponding scenario (divide tensors into multiple small blocks, distinguish data types for parallel computation)
│   │   │   ├── ${op_name}_tiling.h                     # Optional, header file used for Tiling implementation
│   │   │   └── CMakeLists.txt                          # Host-side cmakelist file
│   │   ├── op_api                                      # Optional, operator aclnn implementation file directory, if not configured project will automatically generate
│   │   │   ├── aclnn_${op_name}.cpp                    # Operator aclnn interface implementation file
│   │   │   ├── aclnn_${op_name}.h                      # Operator aclnn interface implementation header file
│   │   │   ├── ${op_name}.cpp                          # Operator l0 interface implementation file
│   │   │   ├── ${op_name}.h                            # Operator l0 interface implementation header file
│   │   │   └── CMakeLists.txt
│   │   │── op_kernel                                   # AI Core operator Device-side Kernel implementation
│   │   │   ├── ${sub_case}                             # Optional, directory used for ${sub_case} sub-scenario
│   │   │   │   ├── ${op_name}_${model}.h               # Operator kernel implementation file, ${model} represents user-defined file name suffix, usually Tiling template name
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # Optional, TilingKey file, defines Key for Tiling strategy, identifies different partitioning methods, if not configured indicates this operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}_tiling_data.h                # Optional, TilingData file, stores Tiling strategy related configuration information, such as block size, parallelism, if not configured indicates this operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}.cpp                          # Kernel entry file, contains main function and scheduling logic
│   │   │   └── ${op_name}.h                            # Kernel implementation file, defines Kernel header file, contains function declarations, structure definitions, logic implementation
│   │   │── op_kernel_aicpu                             # Optional, AI CPU operator Device-side Kernel implementation
│   │   │   ├── ${op_name}_aicpu.cpp                    # Kernel entry file, contains main function and scheduling logic
│   │   │   └── ${op_name}_aicpu.h                      # Kernel header file, contains function declarations, structure definitions, logic implementation
│   │   └── tests                                       # Operator test case directory
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # Optional, UT test cases, develop corresponding cases according to actual situation
│   │           ├── CMakeLists.txt                      # UT case cmakelist file
│   │           ├── op_graph                            # op_graph test case directory
│   │           │   ├── CMakeLists.txt
│   │           │   └── fusion_pass                     # Fusion rule test case directory
│   │           │       └── CMakeLists.txt
│   │           ├── op_host                             # op_host test case directory
│   │           │   ├── CMakeLists.txt
│   │           │   ├── ${op_name}_regbase_tiling.h
│   │           │   ├── op_api                          # op_api test case directory
│   │           │   │   ├── CMakeLists.txt
│   │           │   │   └── test_aclnn_${op_name}.cpp   # Operator aclnn test case file
│   │           │   ├── test_${op_name}_${sub_case}.cpp # op_host test case file in ${sub_case} sub-scenario
│   │           │   ├── test_${op_name}.cpp             # op_host test case file
│   │           │   ├── test_${op_name}_infershape.cpp  # Operator InferShape test case file
│   │           │   └── test_${op_name}_tiling.cpp      # Operator Tiling test case file
│   │           └── op_kernel                           # op_kernel test case directory
│   │               ├── CMakeLists.txt
│   │               │── test_${op_name}.cpp             # Operator Kernel test case file
│   │               └── ${op_name}_data                 # Optional, data comparison and generation scripts depended on in op_kernel test cases, if not configured need to manually implement in corresponding cases
│   │                   ├── compare_data.py             # Data script
│   │                   └── gen_data.py                 # Data generation script
│   └── ...
├── docs                                                # Project related document directory
├── examples                                            # End-to-end operator development and invocation examples
│   ├── add_example                                     # AI Core operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator usage example directory
│   │   ├── op_graph                                    # Operator graph construction related directory
│   │   ├── op_host                                     # Operator information library, Tiling, InferShape related implementation directory
│   │   ├── op_kernel                                   # Operator Kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── add_example_aicpu                               # AI CPU operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator usage example directory
│   │   ├── op_graph                                    # Operator graph construction related directory
│   │   ├── op_host                                     # Operator information library, InferShape related implementation
│   │   ├── op_kernel_aicpu                             # Operator Kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── CMakeLists.txt
│   ├── fast_kernel_launch_example                      # Lightweight, high-performance operator development engineering template
│   │   ├── ascend_ops                                  # Example operator implementation directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── README.md                                   # Lightweight, high-performance operator development engineering documentation
│   │   ├── requirements.txt
│   │   └── setup.py                                    # Build script
│   └── README.md                                       # Project example introduction document
├── scripts                                             # Script directory, contains custom operator, Kernel build related configuration files
├── tests                                               # Project-level test directory
│   ├── requirements.txt                                # Third-party components depended on by test cases
│   └── ut                                              # UT case engineering
│       ├── CMakeLists.txt                              # UT engineering cmakelist script
│       ├── common                                      # Common code used in UT engineering
│       ├── op_api                                      # op_api test engineering
│       ├── op_host                                     # op_host test engineering
│       └── op_kernel                                   # op_kernel test engineering
├── CMakeLists.txt                                      # Project engineering cmakelist entry
├── CONTRIBUTING.md                                     # Project contribution guide file
├── LICENSE                                             # Project open source license information
├── OAT.xml                                             # Configuration script, repository tool usage, used to check if License is standard
├── README.md                                           # Project engineering general introduction document
├── SECURITY.md                                         # Project security declaration file
├── build.sh                                            # Project engineering compilation script
├── classify_rule.yaml                                  # Component division information
├── install_deps.sh                                     # Project install dependency package script
├── requirements.txt                                    # Project third-party dependency packages
└── version.info                                        # Project version information
```
