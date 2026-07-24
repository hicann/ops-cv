# Documentation Contribution Guide

Welcome to contribute to this project's documentation. High-quality documentation is critical to project success. This guide will help you efficiently submit documentation that meets standards.

## Contribution Scope

We welcome any contributions that improve documentation quality, including but not limited to:

- Correction and Improvement: Fix typos, grammar errors, incorrect code examples, outdated information, or broken links.

- Clarification and Optimization: Make descriptions clearer and easier to understand, optimize sentence structure, and supplement background knowledge.

- Content Supplement: Add usage examples, API documentation, frequently asked questions (FAQ), best practices, or warning notes for existing features.

- New Content Creation: Write entirely new chapters or tutorials for new features, such as operator README and API introduction documents. If you have questions, it is recommended to create an Issue for discussion first.

- Localization Translation: Help us translate or proofread documentation in other languages.

- Style and Navigation: Improve the layout, readability, and navigation structure of the documentation website.

## Contribution Process

1. **Preparation**

    - Determine the task: If there are documentation issues, you can create new Issues. The recommended label category is `[Documentation|文档反馈]`, and provide a detailed description. Based on the existing Issues list, determine the documentation issues to be resolved.
    - Claim the task: Comment `/assign @yourself` under the corresponding Issue to indicate that you will handle it and avoid duplicate work.

2. **Document Modification**

    - Select a branch: Download the source code from the master or other Tag branch to your local environment.
    - Follow the format:
      - This project recommends using **Markdown format**.
      - Follow the project's existing writing style.
      - Place static resources such as images in the corresponding directory. For example, images are generally placed in the `figures` folder under the docs directory, and can be adjusted as needed in special cases.
    - Add and delete with caution: When modifying content, try to maintain the original line width and line break conventions.

3. **Submit Changes**

    - Atomic commits: Each commit should focus on one independent modification. For example, "Fix spelling errors in xx guide" and "Update example code in API reference" should be submitted separately.

    - Write clear commit messages:

      ```text
      Brief description (no more than 50 characters)

      If necessary, provide a more detailed description here. Explain the reason and content of the modification, rather than what specifically was changed (the code itself will show that).
      Associated Issue: #123
      ```

4. **Initiate Pull Request**

    - Target branch: Merge the PR into the project's target branch.
    - Title and description:
      - PR title: Should clearly summarize the modification, for example: `[Docs] Fix configuration example in quick start`.
      - PR description: Explain your changes and motivation in detail, and associate the corresponding Issue (use Closes #123 or Fixes #456).
    - Preview check: Check the local or online browsing document effect in advance to ensure rendering meets expectations.
    - Wait for review: Maintainers will review and may propose modification suggestions. Please follow up on the discussion in a timely manner.

## Writing Standards

Before writing project documentation, developers must read the following standards. If you have questions, you are welcome to propose suggestions at any time!

- Prerequisites: First learn the unified writing standards provided by the CANN organization. For details, refer to [CANN Document Writing Standards](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

  - Document content requirements: Introduce required and optional documentation deliverables in the project.
  - Directory structure standards: Introduce the principles of directory division, such as Chinese and English management.
  - Content element standards: Introduce rules for different writing elements, such as file naming, titles, fonts, images, code blocks, and links.

- Precautions:

  In addition to the above writing rules, pay attention to the following:

  - Tone: Use a friendly, professional, and neutral tone. For beginners, avoid unnecessary jargon.
  - Terminology: Maintain terminology consistency (for example, uniformly use "click" instead of "single click"). Refer to the project terminology table (if available).
  - Code examples:
    - Ensure all code examples are runnable and tested.
    - Provide sufficient context and explanation.
    - Indicate the environment or prerequisites required for code execution.
  - Punctuation and format:
    - When mixing Chinese and English, use full-width punctuation. Punctuation marks must conform to the Chinese/English context.
    - Use appropriate heading levels (#, ##, ###).
    - Use lists and tables to organize complex information.
  - Links: Use descriptive link text, avoid "click here", and ensure link resources are authentic and reliable.
  - Images:
    - Common formats: PNG format is recommended, and the style should be consistent with existing images as much as possible.
    - Resolution and clarity: Must be clear and moderately sized, avoiding blurriness or over-compression.
    - File size: Single images are not recommended to exceed 10M.
  - Copyright: Ensure compliance for all referenced images, literature, and other resources.

## Getting Help

If you have any questions during the contribution process:

1. Check existing documentation: If there are problems with templates or standards, first check the project's existing guides, API documentation, or README.
2. Initiate discussion: You can create a new Issue or leave a message directly in the relevant Issue or PR.

## Document Templates

Key documents involved in operator deliverables mainly include the following. For specific writing formats and content requirements, refer to the templates.

- [Operator README Document Template](https://gitcode.com/cann/ops-math/wiki/%E7%AE%97%E5%AD%90README%E6%96%87%E6%A1%A3%E6%A8%A1%E6%9D%BF)
- [aclnn API Document Template](https://gitcode.com/cann/ops-math/wiki/aclnn%20API%E6%96%87%E6%A1%A3%E6%A8%A1%E6%9D%BF)
