# Broadcast Relationship

## Broadcast Concept

Broadcast describes how operators handle tensors (or arrays) of different shapes during operations. In most cases, tensors (or arrays) of different shapes are allowed to automatically expand their shapes during element operations to make their dimensions compatible. Usually, smaller tensors (or arrays) are "broadcast" to larger tensors (or arrays).

Currently, many CANN operator API parameter shapes support broadcasting, which can appropriately improve computational efficiency and reduce memory usage (especially in large-scale data scenarios). For more detailed broadcast technology introduction, refer to the [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) official website.

## Broadcast Rules

Generally, when performing broadcast calculations, you need to understand the following rules:

- Rule 1: If the number of dimensions between arrays is inconsistent, all arrays align with the array with the longest shape. The insufficient part of the shape is filled with 1 on the **left** until the number of dimensions is the same.
  
  > Note:
  > - Example 1: Number of Dimensions refers to the number of dimensions corresponding to the tensor (or array) shape. For example, x.shape=(1,1,2,4), the number of dimensions is 4.
  > - Example 2: For example, calculating a+b, where a.shape=\(2, 2, 3\) and b.shape=\(2, 3\), then array b will be broadcast to b.shape=\(1, 2, 3\).
  
- Rule 2: If the number of dimensions between arrays is consistent, and a certain dimension of an array is 1, then the array with dimension 1 will be stretched to match the corresponding dimension shape of another array.

  > Note:
  > In this scenario, you only need to ensure that broadcasting is performed in a certain dimension. For example, calculating a+b, where a.shape=\(1, 3\) and b.shape=\(3, 1\), then both arrays will be broadcast to a.shape=\(3, 3\) and b.shape=\(3, 3\).

- Rule 3: If the number of dimensions between arrays is inconsistent, and neither has a dimension equal to 1, an error will be reported.

Based on the above rules, the broadcast process generally first expands dimensions according to **Rule 1**, and then stretches the shape according to **Rule 2**. Specific examples are as follows:

```tex
Assume a.shape=(2,2,3), values are like:
[[[1 2 3],[4 5 6]],
 [[1 2 3],[4 5 6]]]
Assume b.shape=(2,3), values are like:
[[1 2 3],
 [-1 -2 -3]]
According to Rule 1, expand dimensions, b.shape=(1,2,3), values are as follows:
[[[1 2 3],
  [-1 -2 -3]]]
According to Rule 2, stretch shape, b.shape=(2,2,3), values are as follows:
[[[1 2 3],[-1 -2 -3]],
 [[1 2 3],[-1 -2 -3]]]
Calculate a+b, actual result is as follows:
 [[[2 4 6],[3 3 3]],
  [[2 4 6],[3 3 3]]]
```

## Limitations

When the data types of two inputs a and b that satisfy the broadcast relationship or the deduced data type are in COMPLEX64, COMPLEX128, DOUBLE, INT16, UINT16, or UINT64, in addition to satisfying the above broadcast rules, the following conditions must also be met. Otherwise, the broadcast will fail, causing the operator execution to report an error.
Condition: The dimension after merging continuous axes that need broadcasting and continuous axes that do not need broadcasting is required to be less than 6.
Examples:

- When a.shape=\(5, 1, 5, 1, 5, 1\) and b.shape=\(5, 5, 5, 5, 5, 5\), there are no axes that need merging. The final dimension is 6, and broadcasting reports an error.
- When a.shape=\(5, 1, 5, 5, 1, 1\) and b.shape=\(5, 5, 5, 5, 5, 5\), broadcasting is not needed in the 2nd and 3rd dimensions, but is needed in the 4th and 5th dimensions. They are merged separately and continuously. The merged dimension is 4, and broadcasting succeeds.
