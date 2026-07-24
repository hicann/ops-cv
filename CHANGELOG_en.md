# CHANGELOG

> This document records important changes in each version, with versions listed in reverse chronological order.

## v8.5.0-beta.1

Release date: 2025-12-30

The first Beta version of ops-math, v8.5.0-beta.1, has been released.
This version introduces multiple new features, bug fixes, and performance improvements, and is currently in the testing phase.
We welcome community feedback to further improve the stability and functionality of ops-math.
For usage instructions, refer to the [official documentation](https://gitcode.com/cann/ops-math/blob/master/README_en.md).

### 🔗 Version Address

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```text
The version directory structure is as follows:
├── aarch64                 # CPU type is ARM
│   ├── ops                  # ops operator package directory for archiving operator subpackages
│   ├── ...
├── x86_64                   # CPU type is X86
│   ├── ops                  # ops operator package directory for archiving operator subpackages
│   ├── ...
```

### 📌 Version Compatibility

**ops-math subpackage and related component compatibility with CANN versions**

| CANN Subpackage Version | Source Code Tag | Compatible CANN Version|
|--|--|--|
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 Key Features

- [Engineering Capability] Support for math ONNX operator plugins. ([#431](https://gitcode.com/cann/ops-math/pull/431))
- [Operator Implementation] Added support for operators concat ([#573](https://gitcode.com/cann/ops-math/pull/573)), lerp ([#519](https://gitcode.com/cann/ops-math/pull/519)), drop_out_v3 ([#539](https://gitcode.com/cann/ops-math/pull/539)), and others in multiple categories.
- [Documentation Optimization] Added a quick start guide and optimized the new operator contribution process in the contribution guide. ([#472](https://gitcode.com/cann/ops-math/pull/472))
- [Usability Improvement] Optimized the kernel compilation process and enabled info-level logging. ([#326](https://gitcode.com/cann/ops-math/pull/326))

### 🐛 Bug Fixes

- The tile operator aclnn interface does not match the operator prototype, causing built-in operator calls to fail. ([Issue239](https://gitcode.com/cann/ops-math/issues/239))
- Operators in the experimental directory report errors when using the repository's built-in op_api for execution. ([Issue143](https://gitcode.com/cann/ops-math/issues/143))
- The operator deployment path does not match the specified vendor_name. ([Issue86](https://gitcode.com/cann/ops-math/issues/86))
- Operator configuration files and folders cannot be created automatically. ([Issue82](https://gitcode.com/cann/ops-math/issues/82))
