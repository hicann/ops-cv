# Documentation Contribution Guide

Welcome to contribute to this project documentation. High-quality documentation is crucial for project success. This guide will help you efficiently submit documentation that meets standards.

## Contribution Scope

We welcome any contributions that improve documentation quality, including but not limited to:

- Correction and Improvement: Fix typos, grammar errors, incorrect code examples, outdated information, or broken links.

- Clarification and Optimization: Make descriptions clearer and easier to understand, optimize sentence structure, and supplement background knowledge.

- Content Supplement: Add usage examples, API documentation, frequently asked questions (FAQ), best practices, or warning notes for existing features.

- New Content Creation: Write entirely new chapters or tutorials for new features, such as operator README and API introduction documents. If you have questions, it is recommended to create an Issue for discussion first.

- Localization Translation: Help us translate or proofread documents in other languages.

- Style and Navigation: Improve the layout, readability, and navigation structure of the documentation website.

## Contribution Process

1. **Preparation**

    - Determine the task: If there are documentation issues, you can create new Issues. It is recommended to use the label category `[Documentation]` and provide a detailed description. Based on the existing Issues list, determine the documentation issue to be resolved.
    - Claim the task: Comment `/assign @yourself` under the corresponding Issue to indicate that you will handle it and avoid duplicate work.

2. **Document Modification**

    - Select a branch: Download the source code from the master or other Tag branches to your local machine.
    - Follow the format:
      - This project recommends using **Markdown format**.
      - Follow the existing writing style of the project.
      - Place static resources such as images in the corresponding directory. For example, images are generally placed in the `figures` folder under the docs directory. You can adjust this for special cases.
    - Add or delete with caution: When modifying content, try to maintain the original line width and line break conventions.

3. **Submit Changes**

    - Atomic commits: Each commit should focus on an independent modification. For example, "Fix spelling errors in xx guide" and "Update example code in API reference" should be submitted separately.

    - Write clear commit messages:

      ```bash
      Brief description (no more than 50 characters)

      If necessary, provide a more detailed description here. Explain the reason and content of the modification, rather than what specifically was changed (the code itself will show that).
      Associated Issue: #123
      ```

4. **Initiate Pull Request**

    - Target branch: Merge the PR into the target branch of the project.
    - Title and description:
      - PR title: Should clearly summarize the modification, for example: `[Docs] Fix configuration example in Quick Start`.
      - PR description: Detailed explanation of your changes, motivation, and associated Issue (use Closes #123 or Fixes #456).
    - Preview check: Check the local or online browsing document effect in advance to ensure rendering meets expectations.
    - Wait for review: Maintainers will review and may provide modification suggestions. Please follow up on the discussion in a timely manner.

## Writing Standards

Before developers write project documentation, be sure to read the following standards first. If you have questions, you are welcome to make suggestions at any time!

- Prerequisites: Please learn the unified writing standards provided by the CANN organization first. For details, refer to the [CANN Document Writing Standards](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

  - Document content requirements: Introduce required and optional documentation deliverables in the project.
  - Directory structure standards: Introduce the principles of directory division, such as Chinese and English management.
  - Content element standards: Introduce rules for different writing elements, such as file naming, titles, fonts, images, code blocks, and links.

- Precautions:

  In addition to the above writing rules, also note the following:

  - Tone: Use a friendly, professional, and neutral tone. For beginners, avoid unnecessary jargon.
  - Terminology: Maintain terminology consistency (such as uniformly using "click" instead of "single click"). Refer to the project terminology table (if available).
  - Code examples:
    - Ensure all code examples are runnable and tested.
    - Provide sufficient context and explanation.
    - Indicate the environment or prerequisites required for code execution.
  - Punctuation and format:
    - When mixing Chinese and English, use full-width punctuation. Punctuation must conform to the Chinese/English context.
    - Use appropriate heading levels (#, ##, ###).
    - Use lists and tables to organize complex information.
  - Links: Use descriptive link text, avoid "click here", and ensure link resources are authentic and reliable.
  - Images:
    - Common format: PNG format is recommended. Try to keep the style consistent with existing images.
    - Resolution and clarity: Must be clear and of moderate size. Avoid blurring or over-compression.
    - File size: It is not recommended for a single image to exceed 10M.
  - Copyright: Ensure compliance for all referenced images, literature, and other resources.

## Get Help

If you have any questions during the contribution process:

1. Check existing documentation: If there are problems with templates or standards, please check the existing guides, API documentation, or README of the project first.
2. Initiate discussion: You can create a new Issue or leave a message directly in the relevant Issue or PR.

## Operator README Template

For `experimental` newly contributed operators, the operator README is a required documentation deliverable. You can refer to the **simple template** provided in this section. You are also supported to expand the content based on this template.

- Document format: Markdown file format is recommended. Support for native or HTML syntax. Please ensure all syntax conforms to official standards.
- Document purpose: Clearly explain operator functions, implementation principles, parameter specifications, and operator invocation methods.
- Section title: Prioritize using template section names (such as Function Description, Parameter Description, etc.). The title level is ##. For special cases, increase the level in order. Support section customization and expansion. Optional sections can be presented as needed.
- Content requirements: For the writing goals and writing standards of each chapter, refer to the detailed description below. For easy understanding, the [AddExample](../examples/add_example/README.md) operator README will be used as an example.

### Product Support Status

> **Writing Standard**: Recommend table format, list supported product models, and mark with √. For product form introduction, refer to [Ascend Product Form Description](https://www.hiascend.com/document/redirect/CannCommunityProductForm).

| Product                                                         | Support Status |
| :----------------------------------------- | :------:|
| Atlas A3 Training Series Products/Atlas A3 Inference Series Products     |    √     |
| Atlas A2 Training Series Products/Atlas A2 Inference Series Products |    √     |

### Function Description

> [!NOTE]
>
> **Writing Goal**: Clarify operator functions, calculation principles, parameter specifications, invocation methods, usage scenarios, etc.
>
> **Writing Standard**: Recommend unordered list format, generally including the following dimensions
>
> - Operator function (required): Please explain the function concisely in one sentence.
> - Calculation formula (optional): Complex functions can use formulas to introduce operator implementation principles or calculation processes in different scenarios.
> - Other dimensions (optional): Support unordered list expansion. Customize according to actual situations, such as calculation examples and flowcharts.

- Operator function: Complete tensor addition calculation.
- Calculation formula:
  $$
  y = x1 + x2
  $$

### Parameter Description

> [!NOTE]
>
> **Writing Goal**: Clarify the meaning, role, specifications, and other information of parameters defined by the operator.
>
> **Writing Standard**: Use table format, generally including the following dimensions
>
> - Parameter name: Explain the parameters in the operator definition file. Keep the order consistent, for example, `op_host/add_example_def.cpp` or `op_graph/add_example_proto.h`.
> - Input/Output/Attribute: Clarify the parameter positioning. Default is required. If optional, it is generally optional input/optional output/optional attribute.
> - Description: Provide parameter meaning, function, usage scenario introduction, including the mapping relationship with the above formula variables.
> - Data type: Data type supported by the parameter. Tensor data type is generally in `DT_XXX` format. For easy writing, the `DT_` prefix can be omitted.
> - Data format: Data layout supported by the parameter. Tensor format is generally in `FORMAT_xxx` format. For easy writing, the `FORMAT_` prefix can be omitted.
> - Other dimensions (optional): Support table field expansion. Customize according to actual situations, such as shape specifications.

|Parameter Name|Input/Output/Attribute|Description|Data Type|Data Format|
|-----|-----------|----|---------|------|
|x1|Input|Represents the first tensor in the add_example calculation, that is, `x1` in the formula.|FLOAT, FLOAT16, INT32|ND|
|x2|Input|Represents the second tensor in the add_example calculation, that is, `x2` in the formula.|Data type consistent with x1|ND|
|y| Output           | Represents the add_example calculation result tensor, that is, `y` in the formula. |FLOAT, FLOAT16, INT32|ND|

### Constraint Description (Optional)

> [!NOTE]
>
> **Writing Goal**: Clarify precautions during operator use, such as parameter combination constraints, applicable scenarios, impact on business, operator performance or accuracy, etc.
>
> **Writing Standard**: **This section is optional**. If there are no constraints, this section content can be omitted; if there are, use unordered list format.

None

### Invocation Description

> [!NOTE]
>
> **Writing Goal**: Provide operator invocation methods. Try to provide sample code that can be directly copied and run for quick verification.
>
> **Writing Standard**: Recommend table format. If the content is complex, other forms can be used.
>
> - Invocation method: Support aclnn, graph mode, and other invocation methods. You can also customize. Please provide at least one method.
> - Sample code: Provide invocation sample code in the operator's `examples` directory, for example, `examples/test_aclnn_add_example.cpp`. The file naming rule is test_\$\{invoke\_mode\}\_\${op_name}. \$\{invoke\_mode\} represents the invocation method, and \${op_name} represents the operator name.
> - Description: Supplementary notes for different invocation methods, such as invocation scenarios, invocation principles, compilation and running guidance, etc. Customize according to actual situations.

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn Invocation</td>
    <td><a href="../examples/add_example/examples/test_aclnn_add_example.cpp">test_aclnn_add_example</a></td>
    <td rowspan="2">Refer to <a href="./zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
  <tr>
    <td>Graph Mode Invocation</td>
    <td><a href="../examples/add_example/examples/test_geir_add_example.cpp">test_geir_add_example</a></td>
  </tr>
</tbody>
</table>

### Reference Resources (Optional)

> [!NOTE]
>
> **Writing Goal**: Provide other supplementary introductions besides operator function, specifications, and invocation, such as operator design documents (Tiling/Kernel design), reference documents, etc.
>
> **Writing Standard**: **This section is optional**. If there are no constraints, this section content can be omitted; if there are, use unordered list format.

None
