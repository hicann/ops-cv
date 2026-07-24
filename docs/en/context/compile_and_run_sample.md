# Compilation and Running Samples

## Prerequisites

- To compile and execute operator APIs, ensure that the basic environment has been set up, including driver, firmware, CANN software package, ops package, etc.
- For operator API invocation process and compilation and running operation details, refer to "Single Operator Invocation > Single Operator API Execution > Calling aclnn Interface Sample Code" in [Application Development (C&C++)](https://hiascend.com/document/redirect/CannCommunityCppInferWizard).

## Preparation Before Compilation

This chapter takes the development and running environment co-location scenario as an example, that is, the machine with AI processor serves as both the development environment and the running environment. In this scenario, code development and code running are on the same machine. Here, the **GridSample operator** is taken as an example. The invocation logic, process, and compilation script of other operators are roughly the same as the GridSample operator. Please modify the API invocation script (\*.cpp) and compilation script (CMakeLists) according to the actual situation.

- **Sample Code**

   It is known that the GridSample operator function is to provide an input tensor and a corresponding grid, and then fill the pixel values at the corresponding positions in the input to the positions specified by the grid according to the coordinate information provided by each position in the grid to obtain the final output. You can get the sample code from "Invocation Example" in [aclnnGridSampler2D](../../../image/grid_sample/docs/aclnnGridSampler2D.md) and name the code file "**test\_grid\_sampler2\_d.cpp**".

- **CMakeLists File**

    The CMake file example is as follows. Please modify according to the actual situation:

    ```CMake
    # Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set project name
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set compilation options
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set executable file name (such as opapi_test) and specify the directory of the operator file *.cpp to be run
    add_executable(opapi_test
                   test_grid_sampler2_d.cpp)

    # Set ASCEND_PATH (CANN software package directory, please modify according to actual path) and INCLUDE_BASE_DIR (header file directory)
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # Set linked library file path
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libacl_rt.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_cv.so)

    # Executable file is in the bin directory under the CMakeLists file directory
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    For operators that combine collective communication and MatMul calculation, collectively referred to as communication-computation fusion operators (abbreviated as MC2 operators), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, MatmulReduceScatter, etc. When calling this type of operator API, it generally involves multi-threading and HCCL (Huawei Collective Communication Library). Therefore, the CMake file needs to additionally import the following content, otherwise compilation will not succeed.

  ```CMake
  # Set linked library file path
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libacl_rt.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_cv.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${CMAKE_THREAD_LIBS_INIT})           # Library file dependent on multi-threading
  ```

  Where "find_package(Threads REQUIRED)" is a CMake command used to find the thread library, which can automatically link the header files or indirectly dependent library files that the thread library depends on.

## Compilation and Running

  1. Prepare the operator invocation code (\*.cpp) and compilation script (CMakeLists.txt) in advance.
  2. Configure environment variables.

    After installing CANN software, log in to the environment using the CANN running user and execute the following command to make the environment variables effective.

        ```sh
        source ${INSTALL_DIR}/set_env.sh
        ```

     Where ${INSTALL_DIR} is the file storage path after CANN software installation. Please replace according to the actual situation.
  3. Compile and run.
        - Enter the directory where CMakeLists.txt is located and execute the following command to create a build directory to store the generated compilation files.

            ```sh
            mkdir -p build
            ```

        - Enter the build directory, execute the cmake command to compile, and then execute the make command to generate the executable file.

          ```sh
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          After successful compilation, the opapi\_test executable file will be generated in the bin folder under the build directory.

        - Enter the bin directory and run the executable file opapi\_test.

        ```sh
        cd bin
        ./opapi_test
        ```

        Taking the GridSample operator running result as an example, the result after running is as follows:

        ```sh
        resultData[0] is: 0.250000
        resultData[1] is: 2.250000
        resultData[2] is: 2.000000
        resultData[3] is: 8.500000
        resultData[4] is: 20.500000
        resultData[5] is: 12.000000
        resultData[6] is: 8.250000
        resultData[7] is: 18.250000
        resultData[8] is: 10.000000
        ```

        If the execution result reports an error and the expected result does not appear, you can use the aclGetRecentErrMsg interface to get the specific error information.
        Calling aclnnGridSampler2DGetWorkspaceSize error to get exception information example is as follows:

        ```sh
        // input is nullptr
        ret = aclnnGridSampler2DGetWorkspaceSize(
        input, grid, interpolationMode, paddingMode, alignCorners, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
        ```

        The above constructing null pointer problem to get error information example is as follows:

        ```sh
        aclnnGridSampler2DGetWorkspaceSize failed. ERROR: 161001
        [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument input.
        ```
