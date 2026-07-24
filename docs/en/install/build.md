# build Parameter Description

## Introduction

build.sh is the build script for this project, located in the project root directory by default. Its purpose is to automatically compile, link and configure source code, ultimately generating executable files, library files or other target files for installation or direct execution. Specifically, the script implements various functions by configuring different parameters, including building multiple target libraries (such as libophost_cv.so), compiling operator packages, executing unit tests, etc.

## Usage

1. **Configure Environment Variables**

   Refer to [Environment Deployment](./quick_install.md) to complete the basic environment setup.

   ```bash
   # Default path installation, using root user as example
   source /usr/local/Ascend/cann/set_env.sh
   ```

2. **Build Command Format**

   Taking the compile operator package command as an example, the format is as follows, where `--vendor_name` and `--ops` are optional in this scenario.

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```

   For full parameter meanings, see the parameter description section below. Please choose appropriate parameters according to actual situations.

## Parameter Description

build.sh supports multiple functions. You can view all function parameters through the following command.

```bash
bash build.sh --help
```

| Parameter Name    | Optional/Required | Parameter Description                                                                 |
|-----------------|-------------|---------------------------------------------------------------------------------|
| -j${n}          | Optional    | Specify the number of compilation threads, ${n} is the specific thread count, default value is 8 (such as -j8); if thread count exceeds CPU core count, it will automatically adjust to CPU core count. |
| -v              | Optional    | View CMake compilation configuration information.                                  |
| -O${n}          | Optional    | Specify compilation optimization level, supports O0/O1/O2/O3 (such as -O3), ${n} is the optimization level identifier. |
| -u              | Optional    | Enable unit test (UT) compilation mode, compile all UT targets.                  |
| --help, -h      | Optional    | Print script usage help information.                                              |
| --ops           | Optional    | Specify operators to compile, such as grid_sample,iou_v2, multiple operators separated by comma ",", cannot be used together with --ophost, --opapi, --opgraph. |
| --soc           | Optional    | Specify NPU model, each compilation only supports 1 NPU model.                   |
| --jit           | Optional    | In static graph scenario, when compiling `cann-${soc_name}-ops-cv_${cann_version}_linux-${arch}.run` package, operator binary files do not need to be compiled (graph runtime will compile online), you can configure this option to improve compilation speed. |
| --static        | Optional    | When configured, indicates generating static library files, including libcann_cv_static.a and aclnn interface header files, combined with --pkg parameter to generate static library compressed package. |
| --vendor_name   | Optional    | Specify the name of custom operator package, default value is custom.            |
| --build-type    | Optional    | Enable debug mode. Optional types: Release/Debug, default is Release. When value is Debug, cannot be used together with --mssanitizer, --oom, --dump_cce. |
| --cov           | Optional    | Reserved parameter, developers do not need to focus on it temporarily.           |
| --noexec        | Optional    | Only compile unit test binary files, do not automatically execute compiled UT executable files. |
| --opkernel      | Optional    | Compile binary kernel.                                                            |
| --pkg           | Optional    | Generate installation package, cannot be used together with -u (UT mode) or --ophost, --opapi, --opgraph. |
| --asan          | Optional    | Enable host-side ASAN (AddressSanitizer) memory detection function.              |
| --valgrind      | Optional    | Reserved parameter, developers do not need to focus on it temporarily.           |
| --make_clean    | Optional    | Execute basic cleanup operation (clean compilation products), script exits after execution. |
| --ophost        | Optional    | Compile libophost_cv.so library, cannot be used together with --pkg, --ops.      |
| --opapi         | Optional    | Compile libopapi_cv.so library, cannot be used together with --pkg, --ops.       |
| --opgraph       | Optional    | Compile libopgraph_cv.so library, cannot be used together with --pkg, --ops.     |
| --ophost_test   | Optional    | Compile ophost related unit tests, equivalent to -u --ophost combination.        |
| --opapi_test    | Optional    | Compile opapi related unit tests, equivalent to -u --opapi combination.          |
| --opgraph_test  | Optional    | Reserved parameter, developers do not need to focus on it temporarily.           |
| --opkernel_test | Optional    | Compile opkernel related unit tests, equivalent to -u --opkernel combination.    |
| --run_example   | Optional    | Compile specified operator and mode examples and execute compiled executable files. |
| --simulator     | Optional    | Enable simulator mode to execute --run_example task. In simulator mode, corresponding simulator library will be linked according to soc_version. |
| --genop         | Optional    | Create AI Core custom operator initial directory.                                 |
| --genop_aicpu   | Optional    | Create AI CPU custom operator initial directory.                                  |
| --experimental  | Optional    | Compile user operators under experimental directory.                              |
| --mssanitizer   | Optional    | Enable kernel-side mssanitizer memory detection function. Cannot be used together with --oom. |
| --oom           | Optional    | Enable kernel-side oom memory detection function. Cannot be used together with --mssanitizer. |
| --dump_cce      | Optional    | Enable kernel-side dump precompiled file function.                                |
| --bisheng_flags | Optional    | Specify BiSheng compiler compilation parameters, multiple compilation parameters separated by comma ",", cannot be used together with --mssanitizer, --oom, --dump_cce. |
| --kernel_template_input  | Optional    | Specify template parameters when compiling kernel, multiple template parameters separated by comma ",", used together with --ops and only one operator can be specified. |
| --cann_3rd_lib_path      | Optional    | Directory where third-party libraries are stored in offline compilation scenario. |
