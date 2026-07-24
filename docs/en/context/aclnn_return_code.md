# aclnn Return Code

When calling aclnn API, common interface return codes are shown in [Table 1](#table1).
For abnormal status code values, you can get exception information through aclGetRecentErrMsg interface (refer to [acl API (C)](https://hiascend.com/document/redirect/CannCommunityCppApi)). You can troubleshoot problems based on error prompts or contact technical support.

**Table 1** Return Status Codes

<a name="table1"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="30.543054305430545%">Status Code Name</th>
<th class="cellrowborder" valign="top" width="15.971597159715973%">Status Code Value</th>
<th class="cellrowborder" valign="top" width="53.48534853485349%">Status Code Description</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="30.543054305430545%">ACLNN_SUCCESS</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%">0</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%">Success.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%">ACLNN_ERR_PARAM_NULLPTR</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%">161001</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%">Parameter validation error, illegal nullptr exists in parameters.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%">ACLNN_ERR_PARAM_INVALID</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%">161002</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%">Parameter validation error, such as two input data types not satisfying input type derivation relationship.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%">ACLNN_ERR_RUNTIME_ERROR</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%">361001</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%">API internal call to npu runtime interface exception.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%">ACLNN_ERR_INNER_XXX</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%">561xxx</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%">API internal exception occurred.</td>
</tr>
</tbody>
</table>

More descriptions about ACLNN_ERR_INNER_XXX class status codes are shown in [Table 2](#table2).

**Table 2** Exception Status Codes

<a name="table2"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="30.183018301830185%">Status Code Name</th>
<th class="cellrowborder" valign="top" width="16.521652165216523%">Status Code Value</th>
<th class="cellrowborder" valign="top" width="53.295329532953296%">Status Code Description</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561000</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: API internal exception occurred.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_INFERSHAPE_ERROR</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561001</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: Error occurred during API internal output shape derivation.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_TILING_ERROR</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561002</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: Exception occurred during API internal npu kernel tiling.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_FIND_KERNEL_ERROR</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561003</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: API internal npu kernel lookup exception (possibly because operator binary package not installed).</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561101</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: API internal aclOpExecutor creation failed (possibly because operating system exception).</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561102</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: API internal did not call uniqueExecutor ReleaseTo.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_NULLPTR</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561103</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, nullptr exception appeared.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561104</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator attribute count exception.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_KEY_CONFILICT</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561105</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Deprecated, please use latest ACLNN_ERR_INNER_KEY_CONFLICT.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_KEY_CONFLICT</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561105</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator kernel matched hash key conflict.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_INVALID_IMPL_MODE</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561106</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator implementation mode parameter error.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561107</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, environment variable ASCEND_OPP_PATH that needs to be configured was not detected.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_LOAD_JSON_FAILED</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561108</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, failed to load operator information json file in operator kernel library.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561109</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, failed to load a field in operator information json file in operator kernel library.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_JSON_FORMAT_INVALID</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561110</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, format field in operator information json file in operator kernel library filled with illegal value.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_JSON_DTYPE_INVALID</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561111</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, dtype field in operator information json file in operator kernel library filled with illegal value.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561112</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator binary kernel library not loaded.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_OP_FILE_INVALID</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561113</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, exception occurred when loading operator json file field.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561114</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator attribute count inconsistent with operator information json, exceeded attr count specified in json.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561115</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator attribute count inconsistent with operator information json, less than attr count specified in json.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561116</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator input count exceeded limit of 32.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_INPUT_JSON_IS_NULL</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561117</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, operator information json file information description missing.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561118</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, exception occurred when parsing workspace information in static binary json file.</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%">ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%">561119</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%">Internal exception: aclnn API internal exception occurred, exception occurred when parsing core count usage information in static binary json file.</td>
</tr>
</tbody>
</table>
