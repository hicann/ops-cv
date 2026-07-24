# Data Format

Data format (format) is used to describe the business semantics of the axes of a multi-dimensional Tensor, representing the physical layout format of data, such as 1D, 2D, 3D, 4D, 5D, etc. Generally, specific formats need to be described in CNN (Convolutional Neural Networks) class APIs.

For the **full range of data formats** supported by aclTensor, refer to "Data Type and Its Operation Interface > aclFormat" in [acl API (C)](https://hiascend.com/document/redirect/CannCommunityCppApi).

For the introduction of **data format layout principles**, refer to "Concept Principles and Terminology > Neural Networks and Operators > Data Layout Format" in [Ascend C Operator Development Guide](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).

## Usage Instructions

Currently, most operator APIs support ND data format. For example, the aclnnAdd interface indicates that the supported data format is ND (that is, the rule of low-dimensional priority continuous layout for multi-dimensional Tensors). For aclnnConvolution, it belongs to CNN class APIs and requires the input aclTensor to be set with a format that has business semantics, rather than ND format. For such operators, the business semantics in the Tensor need to be known during the calculation process to perform the corresponding calculation. For example, in 2D convolution, you need to know the correspondence between the Batch dimension, Channel dimension, Height dimension, Width dimension, and Tensor dimensions.

>**Note:**
>
>- For the parameter description of the two-stage interface, to simplify the description, **the original data format "ACL\_FORMAT\_XXXX" is abbreviated as "\_XXXX\_"**.
>- The meaning of each dimension in the data format: N (Batch) represents batch size, H (Height) represents feature map height, W (Width) represents feature map width, C (Channels) represents feature map channels, D (Depth) represents feature map depth, L (Length) represents feature map length.

## Common Data Formats

When creating an aclTensor through the **aclCreateTensor** interface, you need to set the data format according to the API business requirements. The currently **supported data formats** are:

ACL\_FORMAT\_ND, ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, ACL\_FORMAT\_NDHWC, ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NC, ACL\_FORMAT\_NCL.

For non-ND Tensors, the Tensor dimension requirement is consistent with the format description. For example:

- 5D Tensor: Required to be ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NDHWC, or ACL\_FORMAT\_ND (if the API parameter description does not indicate support for ND, an error will be reported after setting the ND format).
- 4D Tensor: Required to be ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, or ACL\_FORMAT\_ND.
- 3D Tensor: Required to be ACL\_FORMAT\_NCL or ACL\_FORMAT\_ND.
- 2D Tensor: Required to be ACL\_FORMAT\_NC or ACL\_FORMAT\_ND.
- Other dimension Tensors: Required to be ACL\_FORMAT\_ND.

## Private Data Formats

In addition to the above common data formats, there are other data formats, such as ACL\_FORMAT\_NC1HWC0, ACL\_FORMAT\_FRACTAL\_Z, ACL\_FORMAT\_NC1HWC0\_C04, ACL\_FORMAT\_FRACTAL\_NZ, ACL\_FORMAT\_NDC1HWC0, ACL\_FORMAT\_FRACTAL\_Z\_3D, etc.

These formats belong to some private formats of NPU. Currently, most aclnn APIs do not support these formats. If an individual API declares supported data formats, please refer to the actual description of that API.
