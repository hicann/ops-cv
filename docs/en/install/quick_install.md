# Environment Deployment

Before learning [QuickStart](../../QUICKSTART.md) or various [Learning Tutorials](../../../README_en.md#learning-tutorials), please complete the basic environment setup and source code download by referring to the following steps. Ensure that NPU driver, firmware and CANN software (`Ascend-cann-toolkit` and `Ascend-cann-ops`) have been installed.

## Environment Preparation

This project provides multiple ways to set up Ascend environment. Please choose according to your needs.

> **Note**: The meanings of compilation mode and runtime mode mentioned in this document are as follows. Please choose according to actual situations.
>
> - Compilation mode: For scenarios where only compiling this project without running, only need to install CANN toolkit package.
> - Runtime mode: For scenarios where running this project (compile and run or pure run), need to install driver and firmware, CANN toolkit package, CANN ops package.

| Installation Method | Usage Instructions | Usage Scenario |
| ----- | ------ | ------ |
| WebIDE | One-stop development platform, providing online directly runnable Ascend environment, no manual installation required.<br>Currently can provide single machine computing power, **default installation of latest commercial release CANN package**. | Suitable for developers without Ascend devices. |
| Docker | Docker image is an efficient deployment method, pre-integrated with CANN package and necessary dependencies.<br>Currently only applicable to Atlas A2 series products, OS only supports Ubuntu operating system. **Default installation of latest commercial release CANN package**. | Suitable for developers with Ascend devices who need to quickly set up environment. |
| Manual Installation | - | Suitable for developers with Ascend devices who want to experience manual CANN package installation or experience latest master branch capabilities. |

### Method 1: WebIDE Environment

For developers without Ascend devices, you can directly use WebIDE development platform, that is "**Operator One-stop Development Platform**". This platform provides online directly runnable Ascend environment, with necessary driver firmware, software packages and dependencies already installed in the environment, no manual installation required.

> **Note**: The environment defaults to install latest commercial release CANN package. When downloading source code, pay attention to software compatibility. For more introduction about the development platform, please refer to [One-stop Operator Development Tool Platform Operation Guide](https://gitcode.com/org/cann/discussions/54).

1. Enter the open source project, click the "`Cloud Development`" button, and log in with the authenticated Huawei Cloud account. If not registered or authenticated, please follow the page prompts to register and authenticate.
    <!--
   <img src="../figures/cloudIDE.png" alt="Cloud Platform"  width="750px" height="90px">
    -->
2. Follow the page prompts to create and start cloud development environment, click "`Connect>WebIDE`" to enter the operator one-stop development platform. The source code resources of the open source project are by default in the `/mnt/workspace` directory.
<!--
   <img src="../figures/webIDE.png" alt="Cloud Platform"  width="1000px" height="150px">
-->
### Method 2: Docker Deployment

For developers with Ascend devices, if you want to quickly set up Ascend environment, you can use Docker image deployment.

> **Note**:
>
> - Image files are relatively large, downloading takes some time, please wait patiently. For docker command option introduction, you can query through `docker --help`.
> - The environment defaults to install latest commercial release CANN package. When downloading source code, pay attention to software compatibility.

1. **Install Driver and Firmware (Runtime Dependency)**

    For driver and firmware download and installation operations, please refer to "[CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)" in "Prepare Software Packages" and "Install NPU Driver and Firmware" chapters. Driver and firmware are runtime dependencies. If only compiling operators, you can skip installation.

2. **Download Image**

    - Step 1: Log in to the host machine as root user. Ensure Docker engine (version 1.11.2 or above) is installed on the host machine.
    - Step 2: Pull the image pre-integrated with CANN software package and `ops-cv` required dependencies from [Ascend Image Repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884). The command is as follows, choose according to actual architecture:

    ```bash
    # Example: Pull ARM architecture CANN development image
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    # Example: Pull X86 architecture CANN development image
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    ```

3. **Run Docker**

After pulling the image, you need to start the container with specific parameters so that the container can access the host's Ascend device.

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

| Parameter | Description | Notes |
| :--- | :--- | :--- |
| `--name cann_container` | Specify a name for the container for management. | Can be customized. |
| `--device /dev/davinci0` | Core: Map the host's NPU device card to the container, can specify mapping multiple NPU device cards. | Must be adjusted according to actual situation: `davinci0` corresponds to the 0th NPU card in the system. Please execute `npu-smi info` command on the host first, and modify this number according to the device number displayed in the output (such as `NPU 0`, `NPU 1`). |
| `--device /dev/davinci_manager` | Map NPU device management interface. | - |
| `--device /dev/devmm_svm` | Map device memory management interface. | - |
| `--device /dev/hisi_hdc` | Map communication interface between host and device. | - |
| `-v /usr/local/dcmi:/usr/local/dcmi` | Mount device container management interface (DCMI) related tools and libraries. | - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mount `npu-smi` tool. | Enable running this command directly in the container to query NPU status and performance information. |
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: Map the host's NPU driver library to the container. | - |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mount driver version information file. | - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mount CANN software installation information file. | - |
| `-it` | Combination parameter of `-i` (interactive) and `-t` (allocate pseudo terminal). | - |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops` | Specify the Docker image to run. | Please ensure this image name and tag are exactly the same as the image you pulled through `docker pull`. |
| `bash` | Command executed immediately after container starts. | - |

### Method 3: Manual Installation

For developers with Ascend devices, if you want to manually set up Ascend environment, please refer to the following steps.

#### Prerequisites

Please first ensure that the basic library dependencies of the compilation environment have been installed. Note that version number requirements must be met.

- python >= 3.7.0
- gcc >= 7.3.0
- cmake >= 3.16.0
- pigz (optional, can improve packaging speed after installation, recommended version >= 2.4)
- dos2unix
- gawk
- make
- patch
- googletest (only required when executing UT, recommended version [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0))

The above dependencies can be installed one-click through the project root directory install_deps.sh. The command is as follows. If encountering unsupported systems, please refer to this file for self-adaptation.

```bash
bash install_deps.sh
```

After installing the above dependencies, you can continue to install python third-party library dependencies through the project root directory requirements.txt. The command is as follows.

```bash
pip3 install -r requirements.txt
```

#### Software Installation

- **Scenario 1: Experience master version capabilities or develop based on master version**

    1. **Install Driver and Firmware (Runtime Dependency)**

        For download and installation operations, please refer to "[CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)" in "Prepare Software Packages" and "Install NPU Driver and Firmware" chapters. Driver and firmware are runtime dependencies. If only compiling operators, you can skip installation.

    2. **Install CANN Package**

        Please click [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select the latest time version, and download the corresponding package according to product model and environment architecture. The installation command is as follows. For more guidance, refer to "[CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)".

        - Install CANN toolkit package

        ```bash
        # Ensure the installation package has executable permission
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

        - Install CANN ops package (runtime dependency)

        ops package is runtime dependency. If only compiling operators, you can skip installing this package.

        ```bash
        # Ensure the installation package has executable permission
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        - $\{cann\_version\}: Represents CANN package version number.
        - $\{arch\}: Represents CPU architecture, such as aarch64, x86_64.
        - $\{soc\_name\}: Represents NPU model name.
        - $\{install\_path\}: Represents specified installation path. ops package needs to be installed in the same path as toolkit package. Root user default installation is in `/usr/local/Ascend` directory.

- **Scenario 2: Experience released version capabilities or develop based on released version**

    Please visit [CANN Official Download Center](https://www.hiascend.com/cann/download), select release version (only supports CANN 8.5.0 and subsequent versions), and download corresponding package according to product model and environment architecture. Finally refer to the command provided on the webpage to complete installation.

## Environment Verification

After installing CANN package, you need to verify whether the environment and driver are normal.

- **Check NPU Device**

    ```bash
    # Run npu-smi, if device information can be displayed normally, then driver is normal
    npu-smi info
    ```

- **Check CANN Version**

    ```bash
    # View CANN toolkit package version information (default path installation), for WebIDE scenario replace /usr/local with /home/developer
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info
    # View CANN ops package version information (default path installation), for WebIDE scenario replace /usr/local with /home/developer
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

## Environment Variable Configuration

Choose appropriate command as needed to make environment variables take effect.

```bash
# Default path installation, using root user as example (for non-root user, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified path installation
# source ${install_path}/cann/set_env.sh
```

## Source Code Download

Please download corresponding branch source code according to CANN software version. $\{tag\_version\} represents branch tag name. The relationship between branch tag and CANN version refers to [release repository](https://gitcode.com/cann/release-management).

```bash
# Download project corresponding branch source code
git clone -b ${tag_version} https://gitcode.com/cann/ops-cv.git
```

For WebIDE environment, **latest commercial release version project source code is provided by default**. If you need to obtain other version source code, you also need to download source code through the above command.

> [!NOTE] Note
>
> - When using HTTPS protocol on gitcode platform, you need to configure and use personal access token instead of login password for clone, push and other operations.
> - If your compilation environment cannot access the network and cannot download code through git command, please first download source code in networked environment, then manually upload to target environment.
