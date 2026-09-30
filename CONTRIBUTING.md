# Contributing to the PINA Tutorials

Welcome. This document defines the structure, style, and pedagogical philosophy
for every tutorial in this repository. The goal is that all learning materials
are **clear, consistent, pedagogically sound, and beginner-friendly**, while
remaining powerful enough to support advanced use cases.

## Purpose

The tutorials should help users:

- Gain a solid understanding of the PINA library and its core functionality.
- Learn how to work with the PINA modules.
- Explore practical and advanced applications through consistent, hands-on
  code examples.

## Guiding principles

1. **Clarity over cleverness**
   Tutorials should teach, not impress. Prioritise readable, understandable
   code and explanations.

2. **Progressive disclosure of complexity**
   Start simple and introduce complexity gradually. Do not overwhelm readers
   early on.

3. **Consistency is key**
   All tutorials follow the structure below, use the same markdown and code
   formatting, and have a predictable flow.

4. **Real applications, real problems**
   Ground tutorials in real scientific applications or datasets wherever
   possible. Bridge theory and implementation.

## Tutorial structure

### 1. Title

Each tutorial begins with a clear, descriptive title in the format
**Tutorial: TUTORIAL_TITLE**. The title should succinctly communicate the focus
and objective of the tutorial.

### 2. Introducing the topic

Immediately after the title, include a short introduction that outlines the
tutorial's purpose and scope.

- Briefly explain what the tutorial covers and why it is useful.
- Link to relevant research papers, publications, or external resources.
- List the core PINA components or modules that will be used.

If the tutorial assumes knowledge introduced elsewhere, add the
`⚠️ Before starting` admonition pointing at that tutorial.

### 3. Imports and setup

Include a code cell with the necessary setup, so the tutorial runs both locally
and on platforms like Google Colab.

```python
## Routine needed to run the notebook on Google Colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    !pip install "pina-mathlab[tutorial]"

import torch                        # if used
import matplotlib.pyplot as plt     # if used
import warnings                     # if needed

warnings.filterwarnings("ignore")

# Additional PINA and problem-specific imports
...
```

Keep the `Open in Colab` badge as the first cell, so the notebook can be
launched from the rendered documentation page.

### 4. Data generation or loading

- Describe how the data is generated or loaded.
- Comment on data structure, format, and content.
- Visualise key features of the dataset or simulation domain.

If the dataset is small enough to ship, put it in `tutorial<N>/data/`. If it is
downloaded instead, use a `wget` guarded by `IN_COLAB` pointing at a file in
this repository, rather than at a third-party host, so the link cannot rot.

### 5. Main body

The core section presents the problem-solving process in a clear, structured,
pedagogical way. This is where the learning objectives are delivered.

- Guide the reader step-by-step through the PINA workflow.
- Introduce PINA components as they are used.
- Explain the rationale behind modelling decisions.
- Break complex sections with inline comments and markdown explanations.
- Emphasise the relevance of each step to the broader goal.

### 6. Results, visualisation and error analysis

- Show relevant plots of results, for example predicted versus ground truth.
- Quantify performance with metrics such as loss or relative error.
- Discuss the outcomes: strengths, limitations, and any unexpected behaviour.

### 7. What's next?

Every tutorial ends with a **What's Next?** section giving suggestions for
further exploration:

```markdown
## What's Next?

Congratulations on completing the ..., here are a few directions you can explore:

1. **Direction 1** — Suggestion ....

2. **Direction 2** — Suggestion ....

3. **...and many more!** — Other suggestions ....

For more resources and tutorials, check out the [PINA Documentation](https://mathlab.github.io/PINA/).
```

## Writing style

- Use **clear markdown headers** to segment sections.
- Include **inline math** with `$...$` and display math with `$$...$$`.
- Keep paragraphs short and focused.
- Use **bold** and *italic* for emphasis and structure.
- Comment code for clarity.

## Testing tutorials

Every tutorial should:

- Be executable from top to bottom, within the CI time limit (20 minutes per
  notebook). Long-running tutorials should reduce epochs or batch sizes.
- Use only the dependencies declared in this repository's `pyproject.toml`.
- Be committed with its outputs stripped. CI executes the notebook and
  publishes the executed copy to the `built` branch for the documentation
  build.

## Contributing checklist

If you are writing a tutorial:

1. The tutorial follows these guidelines for structure and tone.
2. The tutorial is simple and modular — one tutorial per concept.
3. The notebook's outputs are cleared before committing.
4. The pull request contains only the `.ipynb` file, and the updated
   `README.md`.
5. The pull request targets this repository, not the main PINA repository.

Code style is enforced by `black` (line length 80). The exporter workflow
formats the notebooks for you; to do it locally:

```bash
pip install "black[jupyter]"
black .
```

## Where to file what

| Topic | Repository |
|-------|------------|
| Tutorial content, notebooks, datasets | **this repository** |
| Library behaviour, solvers, models, bugs | [PINA-org/PINA](https://github.com/PINA-org/PINA) |
| Documentation build, rendering | [PINA-org/PINA](https://github.com/PINA-org/PINA) |
