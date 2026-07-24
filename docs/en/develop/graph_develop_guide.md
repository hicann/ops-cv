# Graph Mode Adaptation Guide

## Overview

This document introduces the graph mode adaptation method for custom operators. The overall process is consistent with the operator development guide ([AI Core Operator Development Guide](aicore_develop_guide.md)/[AI CPU Operator Development Guide](aicpu_develop_guide.md)). Notably, **aclnn adaptation is not required**. You only need to complete the following deliverables adaptation to enable graph mode operator invocation.

```Cpp
${op_name}                              # Replace with the lowercase underscore form of the actual operator name
├── op_host                             # Host-side implementation
│   └── ${op_name}_infershape.cpp       # InferShape implementation, implements operator shape derivation, derives output shape at runtime
├── op_graph                            # Graph fusion related implementation
│   ├── CMakeLists.txt                  # op_graph side CMakeLists.txt file
│   ├── ${op_name}_graph_infer.cpp      # InferDataType file, implements operator type derivation, derives output dataType at runtime
└── └── ${op_name}_proto.h              # Operator prototype definition, used for operator recognition during graph optimization and fusion phases
```

This document will use the `AddExample` operator (assuming it is an AI Core operator) as an example to introduce the implementation of graph entry deliverables. AI CPU operator graph entry implementation is basically similar. For complete code, see `add_example` and `add_example_aicpu` under the `examples` directory.

## Shape and DataType Derivation

Graph mode requires completing two deliverables: ```${op_name}_graph_infer.cpp``` and ```${op_name}_infershape.cpp```

**Deliverable 1: ${op_name}_infershape.cpp**

The InferShape function derives the output shape based on the input shape.

The example is as follows. For complete code of `AddExample` operator, please refer to [add_example_infershape.cpp](../../../examples/add_example_aicpu/op_host/add_example_infershape.cpp) under `examples/add_example_aicpu/op_host`.

```C++
// AddExample operator logic is adding two numbers, so output shape is consistent with input shape
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    // Get input shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    // Get output shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    // Get input DimNum
    auto xShapeSize = xShape->GetDimNum();
    // Set output DimNum
    yShape->SetDimNum(xShapeSize);
    // Set input Dim values to output sequentially
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}
// inferShape registration
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample);
```

**Deliverable 2: ${op_name}_graph_infer.cpp**

The InferDataType function derives the output DataType based on the input DataType.

The example is as follows. For complete code of `AddExample` operator, please refer to [add_example_graph_infer.cpp](../../../examples/add_example_aicpu/op_graph/add_example_graph_infer.cpp) under `examples/add_example_aicpu/op_graph`.

```C++
// AddExample operator logic is adding two numbers, so output dataType is consistent with input dataType
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ....
    // Get input dataType
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    // Set input dataType to output
    context->SetOutputDataType(IDX_0, sizeDtype);
    ....
}

// Register InferDataType
IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);
```

## Operator Prototype Configuration

Graph mode invocation requires registering the operator prototype to [Graph Engine](https://www.hiascend.com/cann/graph-engine) (abbreviated as GE), so that GE can recognize the operator's input, output and attribute information. Registration is completed through the `REG_OP` interface. Developers need to define basic information such as operator input, output tensor types and quantities.

Common tensor/attribute data type examples are as follows:

|Tensor Type|Attribute Type|Example|
|-----|------|-----|
|int64|/|DT_INT64|
|int32|/|DT_INT32|
|int16|/|DT_INT16|
|int8|/|DT_INT8|
|double|/|DT_DOUBLE|
|float32|/|DT_FLOAT|
|float16|/|DT_FLOAT16|
|bfloat16|/|DT_BF16|
|complex128|/|DT_COMPLEX128|
|complex64|/|DT_COMPLEX64|
|complex32|/|DT_COMPLEX32|
|/|int|Int|
|/|bool|Bool|
|/|string|String|
|/|float|Float|
|/|list|ListInt|

Basic information is as follows:

|Input/Output|Keyword|Example|
|-----|------|-----|
|Required Input|INPUT|.INPUT(${name}, TensorType({input_dtype}))|
|Optional Input|OPTIONAL_INPUT|.OPTIONAL_INPUT(${name}, TensorType({optional_input_dtype}))|
|Required Attribute|REQUIRED_ATTR|.REQUIRED_ATTR(${name}, ${dtype})|
|Optional Attribute|ATTR|.ATTR(${name}, ${dtype}, ${default_value})|
|Output|OUTPUT|.OUTPUT(${name}, TensorType({output_dtype}))|

Example code is as follows, showing how to register the `AddExample` operator:

```CPP
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
```

For complete code, please refer to [add_example_proto.h](../../../examples/add_example/op_graph/add_example_proto.h) under `examples/add_example/op_graph` directory.
