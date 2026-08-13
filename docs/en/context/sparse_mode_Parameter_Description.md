# sparse_mode Introduction

In the large model field, sparseMode (sparse mode) usually refers to the sparsity design of parameters or activations in the model architecture or calculation formula, which is opposite to the dense mode (DenseMode).

This section introduces commonly used sparseMode and corresponding scenario descriptions.

| sparseMode | Meaning                                  | Note               |
| ---------- | --------------------- | ------------------ |
| 0          | defaultMask mode.                     | -    |
| 1          | allMask mode.                         | -    |
| 2          | leftUpCausal mode.                    | -    |
| 3          | rightDownCausal mode.                 | -    |
| 4          | band mode.                            | -    |
| 5          | prefix non-compressed mode.                    | varlen scenario not supported. |
| 6          | prefix compressed mode.                      | -       |
| 7          | varlen external cut scenario, rightDownCausal mode. | Only varlen scenario supported. |
| 8          | varlen external cut scenario, leftUpCausal mode.    | Only varlen scenario supported. |

The working principle of attenMask is to mask the value of the transpose matrix product of query(Q) and key(K) at the position where Mask is True.
<!-- as shown below:
![Schematic Diagram](../figures/QK转置图.png)
-->
The $QK^T$ matrix will be masked at the position where attenMask is True. 
<!-- with the effect as follows:
![Schematic Diagram](../figures/遮挡QK图.png)
-->
## sparseMode=0

When sparseMode is 0, it represents the defaultMask mode.

- No mask passed: If attenMask is not passed, no mask operation is performed. The attenMask value is None, and preTokens and nextTokens values are ignored. 
<!-- The masked $QK^T$ matrix is shown as follows:

  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵.png)
-->
- nextTokens value is 0, preTokens is greater than or equal to Sq, indicating causal scenario sparse. attenMask should pass a lower triangular matrix. At this time, the part between preTokens and nextTokens needs to be calculated. 
<!-- The masked $QK^T$ matrix is shown as follows:

  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵1.png)
-->
- preTokens is less than Sq, nextTokens is less than Skv, and both are greater than or equal to 0, indicating band scenario. At this time, the part between preTokens and nextTokens needs to be calculated. 
<!-- 

  attenMask should pass a lower triangular matrix, as shown below:
<!--
  ![Schematic Diagram](../figures/attenmask下三角.png)
-->
- preTokens is less than Sq, nextTokens is less than Skv, and both are greater than or equal to 0, indicating band scenario. At this time, the part between preTokens and nextTokens needs to be calculated. 

- preTokens is less than Sq, nextTokens is less than Skv, and both are greater than or equal to 0, indicating band scenario. At this time, the part between preTokens and nextTokens needs to be calculated. 
<!-- The masked $QK^T$ matrix is shown as follows:

  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵2.png)

  attenMask should pass a band-shaped matrix, as shown below:

  ![Schematic Diagram](../figures/attenmask_band形状矩阵.png)
-->
- nextTokens is negative. Taking preTokens=9, nextTokens=-3 as an example, the part between preTokens and nextTokens needs to be calculated. The masked $QK^T$ is shown as follows:

  **Note: When nextTokens is negative, the preTokens value must be greater than or equal to the absolute value of nextTokens, and the absolute value of nextTokens is less than Skv.**
<!--
  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵3.png)
-->
- preTokens is negative. Taking nextTokens=7, preTokens=-3 as an example, the part between preTokens and nextTokens needs to be calculated. The masked $QK^T$ is shown as follows:

  **Note: When preTokens is negative, the nextTokens value must be greater than or equal to the absolute value of preTokens, and the absolute value of preTokens is less than Sq.**
<!--
  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵4.png)
-->
## sparseMode=1

When sparseMode is 1, it represents allMask, that is, passing the complete attenMask matrix.

In this scenario, nextTokens and preTokens values are ignored. 
<!--The masked $QK^T$ matrix is shown as follows:

![Schematic Diagram](../figures/sparsemode为1遮挡矩阵.png)
-->
## sparseMode=2

When sparseMode is 2, it represents the mask of leftUpCausal mode, corresponding to the lower triangular scenario divided by the upper left vertex (parameter starting point is the upper left corner).

In this scenario, preTokens and nextTokens values are ignored. 
<!--The masked $QK^T$ matrix is shown as follows:

![Schematic Diagram](../figures/sparsemode为2遮挡矩阵.png)
-->
The passed attenMask is an optimized compressed lower triangular matrix (2048\*2048).
<!-- The compressed lower triangular matrix is shown as follows (same below):

![Schematic Diagram](../figures/attenmask压缩下三角.png)
-->
## sparseMode=3

When sparseMode is 3, it represents the mask of rightDownCausal mode, corresponding to the lower triangular scenario divided by the lower right vertex (parameter starting point is the lower right corner).

In this scenario, preTokens and nextTokens values are ignored. The attenMask is an optimized compressed lower triangular matrix (2048\*2048). <!--The masked $QK^T$ matrix is shown as follows:
![Schematic Diagram](../figures/sparsemode为3遮挡矩阵.png)
-->
## sparseMode=4

When sparseMode is 4, it represents the band scenario, that is, calculating the part between preTokens and nextTokens. The parameter starting point is the lower right corner, and there must be an intersection between preTokens and nextTokens. The attenMask is an optimized compressed lower triangular matrix (2048\*2048). 
<!--The masked $QK^T$ matrix is shown as follows:

![Schematic Diagram](../figures/sparsemode为4遮挡矩阵.png)
-->
## sparseMode=5

When sparseMode is 5, it represents the prefix non-compressed scenario, that is, on the basis of rightDownCausal, adding a matrix with length Sq and width N on the left. The value of N is obtained from the optional input prefix. For example, the following figure shows prefix passing array [4,5] in batch=2 scenario. The N value of each batch axis can be different. The parameter starting point is the upper left corner.

In this scenario, preTokens and nextTokens values are ignored. The attenMask matrix data format must be BNSS or B1SS. 
<!--The masked $QK^T$ matrix is shown as follows:

![Schematic Diagram](../figures/sparsemode为5遮挡矩阵.png)

The attenMask should pass a matrix as shown below:

![Schematic Diagram](../figures/attenmask矩阵.png)
-->
## sparseMode=6

When sparseMode is 6, it represents the prefix compressed scenario, that is, in the prefix scenario, the attenMask is an optimized compressed lower triangular + rectangular matrix (3072\*2048): the upper part is a [2048, 2048] lower triangular matrix, and the lower part is a [1024, 2048] rectangular matrix. The left half of the rectangular matrix is all 0, and the right half is all 1. The attenMask should pass a matrix as shown below. In this scenario, preTokens and nextTokens values are ignored.
<!--
![Schematic Diagram](../figures/sparsemode为6遮挡矩阵.png)
-->
## sparseMode=7

When sparseMode is 7, it indicates varlen and long sequence external cut scenario (that is, long sequence performs multi-card query sequence length cutting in the model script). Users need to ensure that the scenario using sparseMode 3 is used before external cutting. In the current mode, users need to set preTokens and nextTokens (starting point is the lower right vertex), and need to ensure correct parameters, otherwise there will be accuracy problems.

The masked $QK^T$ matrix is shown as follows. In the second batch, the query is split, and the key and value are not split. The 4x6 mask matrix is split into 2x6 and 2x6 masks, which are calculated on card 1 and card 2 respectively:

- The last block mask of card 1 is a band type mask. Configure preTokens=6 (ensure greater than or equal to the last Skv), nextTokens=-2. actual_seq_qlen should pass {3,5}, and actual_seq_kvlen should pass {3,9}.
- The mask type of card 2 remains unchanged after splitting. sparseMode is 3. actual_seq_qlen should pass {2,7,11}, and actual_seq_kvlen should pass {6,11,15}.
<!--
![Schematic Diagram](../figures/sparsemode为7遮挡矩阵.png)
-->
**Note**:

- sparseMode=7, band represents the sparse type of the last non-empty tensor Batch. If there is only one batch, users need to configure parameters according to the requirements of band mode. When sparseMode=7, users need to input a 2048x2048 lower triangular mask as the input of this fusion operator.
- The sparse parameters of band mode generated based on sparseMode=3 external cutting should meet the following conditions:
  - preTokens >= last_Skv.
  - last_Sq-last_Skv <= nextTokens <= 0.
  - The current mode does not support optional input pse.
- Non-band mode batch should satisfy: Sq <= Skv.

## sparseMode=8

When sparseMode is 8, it indicates varlen and long sequence external cut scenario. Users need to ensure that the scenario using sparseMode 2 is used before external cutting. In the current mode, users need to set preTokens and nextTokens (starting point is the lower right vertex), and need to ensure correct parameters, otherwise there will be accuracy problems.

The masked $QK^T$ matrix is shown as follows. In the second batch, the query is split, and the key and value are not split. The 5x4 mask matrix is split into 2x4 and 3x4 masks, which are calculated on card 1 and card 2 respectively:

- The mask type of card 1 remains unchanged after splitting. sparseMode is 2. actual_seq_qlen should pass {3,5}, and actual_seq_kvlen should pass {3,7}.
- The first block mask of card 2 is a band type mask. Configure preTokens=4 (ensure greater than or equal to the first Skv), nextTokens=1. actual_seq_qlen should pass {3,8,12}, and actual_seq_kvlen should pass {4,9,13}.
<!--
![Schematic Diagram](../figures/sparsemode为8遮挡矩阵.png)
-->
**Note**:

- sparseMode=8, band represents the sparse type of the first non-empty tensor Batch. If there is only one batch, users need to configure parameters according to the requirements of band mode. When sparseMode=8, users need to input a 2048x2048 lower triangular mask as the input of this fusion operator.
- The sparse parameters of band mode generated based on sparseMode=2 external cutting should meet the following conditions:
  - preTokens >= first_Skv.
  - nextTokens >= first_Sq - first_Skv, configure according to the actual situation.
  - The current mode does not support optional input pse.
