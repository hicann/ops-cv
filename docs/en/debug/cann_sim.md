# Introduction

CANN Simulator is a SoC-level chip simulation tool for operator development scenarios. It is used to analyze accuracy and performance data (such as instruction execution status) of AI tasks running on AI simulators at various stages. This tool helps users perform deep performance tuning, enabling developers to obtain verification results and performance feedback almost identical to real chips even when chip resources are unavailable or scarce.

# Main Functions

This tool maintains binary compatibility with on-board execution (the same kernel can be executed on both simulation and AI processor). The main uses are as follows:

* Accuracy simulation: Outputs bit-level accuracy results to help users complete operator accuracy verification.
* Performance simulation: Outputs instruction pipeline diagrams to help users locate operator performance bottleneck issues.

# Preparation Before Use

## Usage Constraints

* Tool recommended environment configuration: CPU 16 cores, memory 32GB or above.
* All example paths in this document need to ensure that the running user has read or read-write permissions.
* For security and minimum privilege considerations, it is recommended to execute this tool with ordinary user privileges. Avoid using high-privilege accounts such as root.
* This tool depends on CANN software package. Before use, please install CANN software package first. No need to install driver and firmware. Execute CANN's set_env.sh environment variable file through source command. For security, please do not modify environment variables involved in set_env.sh after executing source command.
* Users should follow the principle of minimum privilege. For example, files input to the tool should not be writable by other users. In some function scenarios with stricter security requirements, it is also necessary to ensure that input files are not writable by group users.
* This tool is a development tool and is not recommended for use in production environments.
* The simulation function of the tool only supports single-card scenarios and cannot simulate multi-card environments. Only card 0 can be set in the code. Modifying visible card numbers will cause simulation failure.
* The simulation environment only supports AI Core compute operators (does not support MC2 and HCCL type operators).
* CANN Simulator tool is currently in the preview version stage and only supports Ascend950PR chip. It is recommended that the simulator running environment be configured with 16-core CPU and 32GB or more memory.
* Currently arm environment simulation is not supported.

## Environment Preparation

CANN Simulator is integrated in the CANN toolkit package. Refer to [Environment Deployment](../install/quick_install.md) to complete software package installation.

# Quick Start

The following takes [add_examples](../../../examples/add_example) as an example to explain operator simulation in detail.

## Operator Compilation

* Refer to [Operator Invocation](../invocation/quick_op_invocation.md) to complete add_example operator compilation and installation.

```sh
# Note: Enter the project root directory and execute the following compilation command. The command is for reference only. For details, see the operator invocation instructions.
bash build.sh --pkg --soc=Ascend950 --vendor_name=custom --ops=add_example
# Install custom operator package
./build_out/cann-ops-cv-${vendor_name}_linux-${arch}.run
```

* Refer to [aclnn Invocation](../invocation/op_invocation.md#aclnn-invocation) to complete test_aclnn_add_example.cpp compilation and generate executable file test_aclnn_add_example.

## Execute Simulation Command

```sh
cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
```

The simulation tool execution log file is in the examples/add_example/examples/build/bin/cannsim_* directory, and the execution log file is cannsim.log.

From the simulation tool log file, you can see the print information in the sample:

```sh
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
```

## View Performance Pipeline

The simulation performance pipeline file is in the project `examples/add_example/examples/build/bin/cannsim_*/report` directory. The pipeline related files are:

```sh
trace_core0.json
```

Enter "chrome://tracing" address in Chrome browser and drag the generated instruction pipeline file (trace_core0.json) to the blank area to open. For specific parameter introduction, refer to the "Simulation Result Analysis" chapter.

# Simulation Execution Instructions

## Command Function

Execute applications in simulation environment.

## Command Format

cannsim record [options] user_app --user-options

## Parameter Description

Table 1 Simulation Execution Parameter Description

|Parameter|Optional/Required|Description|
| --- | --- | --- |
|-s or --soc-version [options] parameter | Required | Specify the simulation target chip version (such as Ascend950).|
|-o or --output [options] parameter | Optional| The path where generated files are saved. Can be configured as absolute path or relative path. The user executing the tool needs to have read-write permissions. If no path is specified, data is saved in the current directory by default.|
|-g or --gen-report[options] parameter | Optional | Enable whether to perform automatic analysis after simulation completion and generate analysis report. Default is no automatic analysis.|
|user_app|Required|Operator executable file.|
|--user-options|Optional|Operator executable file running parameters.|

## Usage Example

1. Complete operator development and compilation.
2. Execute simulation command. Refer to the following usage example.

    ```sh
    Method 1: Enable simulation and save output to ./output directory. /path/to/app is the operator program.
    $ cannsim record /path/to/app -o ./output -s Ascend950

    Method 2: Enable simulation and generate report for subsequent performance analysis.
    $ cannsim record /path/to/app -o ./output -s Ascend950 --gen-report
    ```

3. After the command completes, a folder named "cannsim_{timestamp}_${user_app}" will be generated in the default path or specified "output" directory. The structure example is as follows:

    ```cpp
    ├─cannsim_{timestamp}_${user_app}
    ├── cannsim.log
    ```

4. Users can obtain operator execution results and perform accuracy comparison. Results are displayed in cannsim.log. The example is as follows.

    The following output is only an example of AscendC single operator direct invocation accuracy comparison result. There may be slight differences due to different versions. Please refer to the actual output.

    ```sh
    INFO:root:[INFO] compare data case[ case001]
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:[' case001', 0, 65536, 'Success']
    ```

5. View operator instruction pipeline diagram. Refer to simulation result analysis.

# Simulation Result Analysis Instructions

## Command Function

Generate visualized instruction pipeline diagram.

## Command Format

cannsim report [options]

## Parameter Description

Table 1 Simulation Result Analysis Parameter Description

|Parameter | Optional/Required | Description|
| --- | --- | --- |
|-e or --export [options] parameter | Required | Original result file directory. Need to specify the result directory generated after simulation execution. Specify to cannsim_{timestamp}_${user_app} level. Can be configured as absolute path or relative path. The user executing the tool needs to have read-write permissions.|
|-o or --output [options] parameter | Optional | Analysis result output directory. Can be configured as absolute path or relative path. The executing user needs to have read-write permissions. If no path is specified, data is saved in the current directory by default. If the generated result file has the same name as an existing file, the original file will be overwritten.|
|-n or --core-id [options] parameter | Optional | Specify the core ID for generating instruction pipeline. If not specified, pipeline for core 0 is generated by default. Configuration format is as follows: To generate pipeline for all cores, configure as 'all'. Specify core ID range, such as '0-1'. Specify single core ID, such as '5'.|

## Usage Example

1. Refer to simulation execution to execute operator simulation. Compare output examples to ensure corresponding results execute correctly.
2. Execute simulation result analysis command. Refer to the following execution use case.

    ```sh
    Generate performance analysis report in current directory (default analysis of core 0 only)
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app}

    Generate performance analysis reports for core 0, core 1, core 11, core 12 in specified directory
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} -o /path/to/report -n '0-1, 11-12'
    ```

3. After the command executes, corresponding pipeline files will be generated in the output configured directory. The file format is json format. The output result example is as follows:

    ```sh
    trace_core0.json
    trace_core1.json
    ...
    ```

4. View simulation results
    Enter "chrome://tracing" address in Chrome browser and drag the generated instruction pipeline file (trace.json) to the blank area to open. 
    <!--
    You can use keyboard shortcuts (W: zoom in, S: zoom out, A: move left, D: move right) to view.
    ![Instruction Pipeline Diagram](../figures/Instruction_Pipeline_Diagram.png)
    -->
    Table 2 Key Field Description

    |Field Name|Field Meaning|
    | --- | --- |
    |VECTOR|Vector computation unit.|
    |SCALAR|Scalar computation unit.|
    |Cube|Matrix multiplication computation unit.|
    |MTE1|Data transfer pipeline. Data transfer direction is: L1 ->{L0A/L0B, UBUF}.|
    |MTE2|Data transfer pipeline. Data transfer direction is: {DDR/GM, L2} ->{L1, L0A/B, UBUF}.|
    |MTE3|Data transfer pipeline. Data transfer direction is: UBUF -> {DDR/GM, L2, L1}, L1->{DDR/L2}.|
    |FIXP|Data transfer pipeline. Data transfer direction is: FIXPIPE L0C -> OUT/L1.|
    |FLOWCTRL|Control flow instruction.|
    |ICACHELOAD|View missed ICache.|

# Query Help Information

## Command Function

Query tool help information.

## Command Format

Query tool help information:

```sh
cannsim --help
```

Query tool record subcommand help information:

```sh
cannsim record --help
```

Query tool report subcommand help information:

```sh
cannsim report --help
```

## Parameter Description

None

## Usage Example

1. Log in to Host server.
2. Execute the following command.

    ```sh
    cannsim --help
    ```

## Output Description

```sh
usage: cannsim [-h] {record,report} ...

Command-line tool for performance simulation analysis on Ascend hardware.

positional arguments:
  {record,report}  Available commands
    record         Run user application in AscendOps simulation environment
    report         Generate performance analysis reports

options:
  -h, --help       show this help message and exit
```
