# Security Statement

## Running User Recommendations

From a security perspective, it is not recommended to use root or other administrator-type accounts to execute any commands. Follow the principle of least privilege.

## File Permission Control

- It is recommended that users set the running system umask value to 0027 or higher on the host (including the host machine) and in containers to ensure that new folders have a default maximum permission of 750 and new files have a default maximum permission of 640.
- It is recommended that users implement permission control and other security measures for sensitive content such as personal privacy data, business assets, source files, and various files saved during operator development. For example, for this project's installation directory permission control and input public data file permission control, refer to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario).
- During operator runtime, operator compilation files may be cached and stored in the `kernel_meta_*` folder in the running directory to accelerate subsequent operator calls. Users can perform permission control on the generated related files as needed.
- Users should implement permission control during installation and use. For file permission references, refer to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario).

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. Some intermediate files will be generated during the compilation process. It is recommended that you implement permission control for intermediate files after compilation to ensure file security.

## Runtime Security Statement

- It is recommended that users write corresponding operator invocation scripts based on the runtime environment resource conditions. If the operator invocation script does not match the resource conditions, such as when the space used for generating input data or reference calculation results exceeds the memory capacity limit, or when the script saves data locally exceeding the disk space size, errors may occur and cause the process to exit unexpectedly.
- When an operator runs abnormally, it will exit the process and print error information. It is recommended to locate the specific error cause based on the error prompt, including setting operator synchronous execution and viewing log files.
- When operators are invoked through [PyTorch](https://gitee.com/ascend/pytorch), runtime errors may occur due to version mismatch. For details, refer to the [PyTorch Security Statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network addresses contained in this project's code are as follows:

| Type | Open Source Code Address | File Name | Public Network IP Address/Public Network URL Address/Domain Name/Email Address/Compressed File Address | Purpose Description |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
|  Dependency  | Not applicable  | cmake/third_party/makeself-fetch.cmake | [https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) | Download makeself source code from gitcode as a compilation dependency |
|  Dependency  | Not applicable  | cmake/third_party/json.cmake | [https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) | Download json source code from gitcode as a compilation dependency |
|  Dependency  | Not applicable  | cmake/third_party/gtest.cmake | [https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) | Download googletest source code from gitcode as a compilation dependency |
|  Dependency  | Not applicable  | cmake/third_party/eigen.cmake | [https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) | Download eigen source code from gitcode as a compilation dependency |

---

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario

| Type           | Linux Permission Reference Maximum Value |
| -------------- | ---------------  |
| User home directory                        |   750 (rwxr-x---)            |
| Program files (including script files, library files, etc.)       |   550 (r-xr-x---)             |
| Program file directory                      |   550 (r-xr-x---)            |
| Configuration file                          |  640 (rw-r-----)             |
| Configuration file directory                      |   750 (rwxr-x---)            |
| Log file (recording completed or archived)        |  440 (r--r-----)             |
| Log file (currently recording)                |    640 (rw-r-----)           |
| Log file directory                      |   750 (rwxr-x---)            |
| Debug file                         |  640 (rw-r-----)         |
| Debug file directory                     |   750 (rwxr-x---)  |
| Temporary file directory                      |   750 (rwxr-x---)   |
| Maintenance upgrade file directory                  |   770 (rwxrwx---)    |
| Business data file                      |   640 (rw-r-----)    |
| Business data file directory                  |   750 (rwxr-x---)      |
| Key component, private key, certificate, ciphertext file directory    |  700 (rwx-----)      |
| Key component, private key, certificate, encrypted ciphertext        | 600 (rw-------)      |
| Encryption/decryption interface, encryption/decryption script            |   500 (r-x------)        |
