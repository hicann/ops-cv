# Operator Invocation Methods

## Overview

Operators can be invoked through multiple methods. This chapter uses the `AddExample` operator invocation as an example to introduce operator invocation and execution process in detail.

> Note: aclnn interface invocation is recommended first. If this method is not supported, you can use graph construction method to invoke operators.

- aclnn invocation **(Recommended)**: Invoke operators through operator aclnnXxx interface (a set of C-based APIs, no IR definition required).
- Graph mode invocation: Invoke operators through IR (Intermediate Representation) graph construction method.

## aclnn Invocation

### Invocation Flow
<!--
![Schematic Diagram](../figures/aclnn调用.png)
-->

### Example Code

The example code for invoking `AddExample` operator through aclnn interface is as follows (detailed code see [test_aclnn_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp)), **for reference only**. Other operator interface invocation processes are similar, please replace with actual aclnn interface. Before invocation, please set environment variables according to the environment installation prompt information.

Note: If you need to invoke other operators in this project, you can access test\_aclnn\_\$\{op\_name\}.cpp under the corresponding operator's `examples` directory, where $\{op\_name\} represents operator name.

```Cpp
int main()
{
    // 1. Call acl for device/stream initialization
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct input and output, need to customize construction according to API interface
    aclTensor* selfX = nullptr;
    void* selfXDeviceAddr = nullptr;
    std::vector<int64_t> selfXShape = {32, 4, 4, 4};
    std::vector<float> selfXHostData(2048, 1);
    ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, aclDataType::ACL_FLOAT, &selfX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* selfY = nullptr;
    void* selfYDeviceAddr = nullptr;
    std::vector<int64_t> selfYShape = {32, 4, 4, 4};
    std::vector<float> selfYHostData(2048, 1);
    ret = CreateAclTensor(selfYHostData, selfYShape, &selfYDeviceAddr, aclDataType::ACL_FLOAT, &selfY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {32, 4, 4, 4};
    std::vector<float> outHostData(2048, 1);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call CANN operator library API, need to modify to specific Api name
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. Call aclnnAddExample first segment interface
    ret = aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 5. Apply for device memory according to workspaceSize calculated by first segment interface
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 6. Call aclnnAddExample second segment interface
    ret = aclnnAddExample(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExample failed. ERROR: %d\n", ret); return ret);

    // 7. (Fixed writing) Synchronously wait for task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 8. Get output value, copy result from device side memory to host side, need to modify according to specific API interface definition
    PrintOutResult(outShape, &outDeviceAddr);

    // 9. Release aclTensor, need to modify according to specific API interface definition
    aclDestroyTensor(selfX);
    aclDestroyTensor(selfY);
    aclDestroyTensor(out);

    // 10. Release device resources
    aclrtFree(selfXDeviceAddr);
    aclrtFree(selfYDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 11. acl deinitialization
    aclFinalize();
    return 0;
}
```

### Compilation and Running

> Note: For operators already implemented in the project (non-custom operators), you can directly run operators through [build.sh](../../../build.sh) in the root directory. For operations, please refer to [Local Verification](./quick_op_invocation.md#local-verification).

1. Prerequisites.
   Please refer to this project [Source Code Compilation](./quick_op_invocation.md#source-code-compilation) to complete target operator compilation and deployment.
   The currently provided CMakeLists.txt example is only for custom operator package scenario. After custom operator package installation, please set corresponding environment variables according to installation prompts.

2. Create CMakeLists.txt file.

   Create CMakeLists.txt file in the same directory as test\_aclnn\_\$\{op\_name\}.cpp. Taking `AddExample` operator as an example, the example is as follows. Please modify according to actual situation.

    ```bash
   cmake_minimum_required(VERSION 3.14)
   # Set project name
   project(ACLNN_EXAMPLE)

   # Set C++ compilation standard
   add_compile_options(-std=c++11)

   # Set compilation output directory to bin folder in current directory
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

   # Set compilation options for debug and release modes
   set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
   set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

   # Get LD_LIBRARY_PATH environment variable
   if(NOT DEFINED ENV{LD_LIBRARY_PATH})
       message(FATAL_ERROR "LD_LIBRARY_PATH environment variable is not set")
   endif()
   set(LD_LIB_PATH "$ENV{LD_LIBRARY_PATH}")

   # Split path list and find path containing /vendors/ (only needed for custom operators)
   string(REPLACE ":" ";" LD_LIB_LIST "${LD_LIB_PATH}")
   set(TARGET_PATH "")
   foreach(path ${LD_LIB_LIST})
       # Match path containing /vendors/ (position insensitive)
       if(path MATCHES "/vendors/")
           set(TARGET_PATH "${path}")
           break()
       endif()
   endforeach()
   if(NOT TARGET_PATH)
       message(FATAL_ERROR "Path containing /vendors/ not found in LD_LIBRARY_PATH")
   endif()
   if(TARGET_PATH MATCHES "/vendors/([^/]+)")
       set(TARGET_SUBDIR "${CMAKE_MATCH_1}")
   else()
       message(FATAL_ERROR "Direct subdirectory of /vendors/ not found in path ${TARGET_PATH}")
   endif()

   # Add executable file (please replace with actual operator executable file), specify operator invocation *.cpp file
   add_executable(test_aclnn_add_example
   test_aclnn_add_example.cpp)

   # ASCEND_PATH (CANN software package directory, please modify according to actual path)
   if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
       set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
   else()
       set(ASCEND_PATH "/usr/local/Ascend/cann")
   endif()

   # Set header file path
   set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
   include_directories(
       ${INCLUDE_BASE_DIR}
       ${ASCEND_PATH}/opp/vendors/${TARGET_SUBDIR}/op_api/include    # Only needed for custom operators
       # ${INCLUDE_BASE_DIR}/aclnn                                   # Only needed for built-in operators
   )
   include_directories(
       ${INCLUDE_BASE_DIR}
   )

   # Link required dynamic libraries
   target_link_libraries(test_aclnn_add_example PRIVATE             # Replace with actual operator executable file
       ${ASCEND_PATH}/lib64/libascendcl.so
       ${ASCEND_PATH}/lib64/libnnopbase.so
       ${ASCEND_PATH}/opp/vendors/${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so   # Only needed for custom operators
       # ${ASCEND_PATH}/lib64/libopapi_cv.so    # Only needed for built-in operators
   )

   # Install target files to bin directory
   install(TARGETS test_aclnn_add_example DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

3. Create run.sh file.

    Create run.sh file in the same directory as test\_aclnn\_\$\{op\_name\}.cpp. Taking `AddExample` operator as an example, the example is as follows. Please modify according to actual situation.

    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                      # Actual CANN package installation path
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

    rm -rf build
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # Execute build command
    make
    cd bin
    ./test_aclnn_add_example            # Replace with actual operator executable file name
    ```

4. Run run.sh file.
    Execute the following command in the path where run.sh file is located:

   ```bash
   bash run.sh
   ```

    By default, executable file test\_aclnn\_add\_example is generated in current execution path `/build/bin`. The running result is as follows:

   ```sh
   mean result[2046] is 2.000000
   mean result[2047] is 2.000000
   ```

## Graph Mode Invocation

### Invocation Flow
<!--
![Schematic Diagram](../figures/IR调用.png)
-->

### Example Code

The example code for invoking `AddExample` operator through graph method is as follows (detailed code see [test_geir_add_example.cpp](../../../examples/add_example/examples/test_geir_add_example.cpp)), **for reference only**. Other operator invocation processes are similar, please replace with actual operator prototype.

If you need to invoke other operators in this project, you can access test\_geir\_\$\{op\_name\}.cpp under the corresponding operator's `examples` directory, where $\{op\_name\} represents operator name.

```CPP
int main() {
    // 1. Create graph object
    Graph graph(graphName);

    // 2. Graph global compilation option initialization
    Status ret = ge::GEInitialize(globalOptions);

    // 3. Create AddExample operator instance
    auto add1 = op::AddExample("add1");

    // 4. Define graph input and output vectors
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    // 5. Prepare input data
    std::vector<int64_t> xShape = {32,4,4,4};
    // Macro expansion method to handle variable assignment
    ADD_INPUT(1, x1, inDtype, xShape);
    ADD_INPUT(2, x2, inDtype, xShape);
    ADD_OUTPUT(1, y, inDtype, xShape);

    outputs.push_back(add1);

    // 6. Set graph object input operators and output operators
    graph.SetInputs(inputs).SetOutputs(outputs);

    // 7. Create session object
    ge::Session* session = new Session(buildOptions);

    // 8. Session add graph
    ret = session->AddGraph(graphId, graph, graphOptions);

    // 9. Run graph
    ret = session->RunGraph(graphId, input, output);

    // 10. Release resources
    GEFinalize();

    return 0;
}
```

### Compilation and Running

> Note: For operators already implemented in the project (non-custom operators), you can directly run operators through [build.sh](../../../build.sh) in the root directory. For operations, please refer to [Local Verification](./quick_op_invocation.md#local-verification).

1. Prerequisites.
   Please refer to this project [Source Code Compilation](./quick_op_invocation.md#source-code-compilation) to complete target operator compilation and deployment.

2. Create CMakeLists.txt file.

   Create CMakeLists.txt file in the same directory as test\_geir\_\$\{op\_name\}.cpp. Taking `AddExample` operator as an example, the example is as follows. Please modify according to actual situation.

    ```bash
   cmake_minimum_required(VERSION 3.14)

   # Set project name
   project(GE_IR_EXAMPLE)

   if(NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
       get_filename_component(ASCEND_PATH $ENV{ASCEND_OPP_PATH} DIRECTORY)
   elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
       set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
   else()
       set(ASCEND_PATH "/usr/local/Ascend/cann")
   endif()

   set(FWK_INCLUDE_DIR "${ASCEND_PATH}/compiler/include")

   message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

   file(GLOB files CONFIGURE_DEPENDS
        test_geir_add_example.cpp
   )

   # Add executable file (please replace with actual operator executable file)
   add_executable(test_geir_add_example ${files})

   find_library(GRAPH_LIBRARY_DIR libgraph.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GE_RUNNER_LIBRARY_DIR libge_runner.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GRAPH_BASE_LIBRARY_DIR libgraph_base.so "${ASCEND_PATH}/compiler/lib64")

   # Link required dynamic libraries
   target_link_libraries(test_geir_add_example PRIVATE
        ${GRAPH_LIBRARY_DIR}
        ${GE_RUNNER_LIBRARY_DIR}
        ${GRAPH_BASE_LIBRARY_DIR}
   )

   # Set header file path
   target_include_directories(test_geir_add_example PRIVATE
        ${FWK_INCLUDE_DIR}/graph/
        ${FWK_INCLUDE_DIR}/ge/
        ${ASCEND_PATH}/opp/built-in/op_proto/inc/
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_PATH}/compiler/include
   )
    ```

3. Create run.sh script.

   Create run.sh file in the same directory as test\_geir\_\$\{op\_name\}.cpp. Taking `AddExample` operator as an example, the example is as follows. Please modify according to actual situation.

    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                      # Actual CANN package installation path
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

    rm -rf build
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # Execute build command
    make
    ./test_geir_add_example                  # Replace with actual operator executable file name
    ```

4. Run run.sh script.
    Execute the following command in the path where run.sh file is located:

    ```bash
    bash run.sh
    ```

    By default, executable file test\_geir\_add\_example is generated in current execution path `/build/bin`. The running result is as follows:

    ```sh
    INFO - [XIR]: Finalize ir graph session success
    ```
