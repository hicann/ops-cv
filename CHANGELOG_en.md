# CHANGELOG

> This document records important changes in each version, with versions listed in reverse chronological order.

## v8.5.0-beta.1

Release Date: 2025-12-30

The first Beta version v8.5.0-beta.1 of ops-cv operators is now released.
This version introduces multiple new features, bug fixes, and performance improvements, and is currently in the testing phase.
We sincerely welcome community feedback to further improve the stability and functional completeness of ops-cv.
For usage instructions, refer to the [official documentation](https://gitcode.com/cann/ops-cv/blob/9.0.0/README_en.md).

### 🔗 Version Address

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```Cpp
The version directory structure is as follows:
├── aarch64                 # CPU type is ARM
│   ├── ops                  # ops operator package directory, used for archiving operator subpackages
│   ├── ...
├── x86_64                   # CPU type is X86
│   ├── ops                  # ops operator package directory, used for archiving operator subpackages
│   ├── ...
```

### 📌 Version Compatibility

**ops-cv subpackage and related components compatibility with CANN versions**

| CANN Subpackage Version | Version Source Code Tag | Compatible CANN Version|
|--|--|--|
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 Key Features

- [Engineering Capability] CV-class ONNX operator plugin support. ([#97](https://gitcode.com/cann/ops-cv/pull/97))
- [Engineering Capability] Added compilation options such as oom, asan, mssanitizer, and build-type for engineering-level stability and debuggability. ([#62](https://gitcode.com/cann/ops-cv/pull/62), [#70](https://gitcode.com/cann/ops-cv/pull/70), [#80](https://gitcode.com/cann/ops-cv/pull/80))
- [Operator Implementation] Image sampling and processing operators added support for KirinX90. ([#105](https://gitcode.com/cann/ops-cv/pull/105), [#125](https://gitcode.com/cann/ops-cv/pull/125))
- [Operator Implementation] The three_interpolate_backward operator added deterministic computation feature. ([#39](https://gitcode.com/cann/ops-cv/pull/39))
- [Documentation Optimization] Added QUICK_START, offline compilation mode, and improved development guides for aicore/aicpu/graph modes. ([#40](https://gitcode.com/cann/ops-cv/pull/40), [#156](https://gitcode.com/cann/ops-cv/pull/156))
- [Documentation Optimization] Optimized the new operator contribution process in the contribution guide. ([#57](https://gitcode.com/cann/ops-cv/pull/57))
- [Performance Optimization] Added asc_opc operator parallel compilation capability to optimize compilation efficiency; added ccache to optimize compilation time. ([80](https://gitcode.com/cann/ops-cv/pull/80))

### 🐛 Bug Fixes

- Fixed the add_example sample operator execution invocation issue. ([Issue16](https://gitcode.com/cann/ops-cv/issues/16))
- Fixed the issue where compiling non-existent operators showed compilation success. ([Issue25](https://gitcode.com/cann/ops-cv/issues/25))
- Used constexpr to modify if for compilation optimization. ([Issue32](https://gitcode.com/cann/ops-cv/issues/32))
