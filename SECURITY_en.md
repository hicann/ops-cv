# Security Statement

## Running User Recommendations

From a security perspective, it is not recommended to use administrator accounts such as root to execute any commands. Follow the principle of least privilege.

## File Permission Control

- It is recommended that users set the running system umask value to 0027 or above on the host (including the host machine) and in the container to ensure that the default maximum permission for new folders is 750 and the default maximum permission for new files is 640.
- It is recommended that users implement security measures such as permission control for sensitive content such as personal privacy data, business assets, source files, and various files saved during operator development. For example, for project installation directory permission control and input public data file permission control, the recommended permissions refer to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario).
- During operator runtime, operator compilation files may be cached and stored in the `kernel_meta_*` folder in the running directory to speed up subsequent operator invocation. Users can perform permission control on the generated related files as needed.
- Users need to implement permission control during installation and use. It is recommended to refer to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario) for file permission settings.

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. Some intermediate files will be generated during the compilation process. It is recommended that you implement permission control for the intermediate files after compilation to ensure file security.

## Runtime Security Statement

- It is recommended that users write corresponding operator invocation scripts based on the running environment resource status. If the operator invocation script does not match the resource status, such as when the space used for generating input data or benchmark calculation results exceeds the memory capacity limit, or when the script saves data locally exceeding the disk space size, errors may occur and cause the process to exit unexpectedly.
- When the operator runs abnormally, it will exit the process and print error information. It is recommended to locate the specific error cause based on the error prompt, including setting operator synchronous execution and viewing log files.
- When operators are invoked through [PyTorch](https://gitee.com/ascend/pytorch), runtime errors may occur due to version mismatch. For details, refer to the [PyTorch Security Statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network addresses contained in this project code are shown as follows:

|      Type      |                                           Open Source Code Address                                           |                            File Name                             |             Public Network IP Address/Public Network URL Address/Domain Name/Email Address/Compressed File Address             |                   Usage Description                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
|  Dependency  | Not applicable  | cmake/third_party/makeself-fetch.cmake | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) | Download makeself source code from gitcode, serving as compilation dependency |
|  Dependency  | Not applicable  | cmake/third_party/json.cmake | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) | Download json source code from gitcode, serving as compilation dependency |
|  Dependency  | Not applicable  | cmake/third_party/gtest.cmake | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) | Download googletest source code from gitcode, serving as compilation dependency |
|  Dependency  | Not applicable  | cmake/third_party/eigen.cmake | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) | Download eigen source code from gitcode, serving as compilation dependency |

---

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario

| Type               | Linux Permission Reference Maximum Value |
|------------------| ---------------  |
| User Home Directory            |   750 (rwxr-x---)            |
| Program Files (including script files, library files, etc.) |   550 (r-xr-x---)             |
| Program File Directory           |   550 (r-xr-x---)            |
| Configuration File             |  640 (rw-r-----)             |
| Configuration File Directory           |   750 (rwxr-x---)            |
| Log File (recording completed or archived) |  440 (r--r-----)             |
| Log File (currently recording)      |    640 (rw-r-----)           |
| Log File Directory           |   750 (rwxr-x---)            |
| Debug File          |  640 (rw-r-----)         |
| Debug File Directory        |   750 (rwxr-x---)  |
| Temporary File Directory           |   750 (rwxr-x---)   |
| Maintenance Upgrade File Directory         |   770 (rwxrwx---)    |
| Business Data File           |   640 (rw-r-----)    |
| Business Data File Directory         |   750 (rwxr-x---)      |
| Key Component, Private Key, Certificate, Ciphertext File Directory |  700 (rwx------)      |
| Key Component, Private Key, Certificate, Encrypted Ciphertext  | 600 (rw-------)      |
| Encryption/Decryption Interface, Encryption/Decryption Script      |   500 (r-x------)        |
