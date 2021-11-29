# Table of contents
1. [Coding standards](#coding-standards)
1. [Working with repositories](#working-with-repositories)

# Coding standards

1. [Code standard](#code-standard)
1. [Comment standard](#comment-standard)
1. [Spaces vs Tabs](#spaces-vs-tabs)
1. [Imports](#imports)

## Code standard

Since it is the most popular and well established coding standard a [PEP 8](https://www.python.org/dev/peps/pep-0008/)  was chosen as a go-to reference. The compatibility of code with PEP 8 standard can be checked through the use of [flake8 library](https://flake8.pycqa.org/en/latest/)

## Comment standard

Due to the project having numerical nature for the compatibility of convention with the used library, the [numpy comment style](https://numpydoc.readthedocs.io/en/latest/format.html).

### What to comment

Every function containing a functionality connected with the project's implementation and function whose implementation/procedure, after reading the function name and having a glance at the code, is still ambiguous.

Example of a function not worth commenting:

```python
def add(a: int, b: int)
  eturn a + b
```

## Spaces vs Tabs

As in the rest of the repository.

## Imports

The standard import structure with the import order centring around the purpose of the library.
Examples:

```python
import os
import sys
```

```python
from subprocess import Popen, PIPE
```

The imports should contain specific imported classes if possible, going one module deeper is only acknowledged if the name conflict would appear.



Additionally, importing all submodules from a module using "\*" character is strictly prohibited. Example of such code can be seen below:

```
from numpy import *
```

All imports of different modules that cover different use-cases should be divided by an empty line. Example

```
import numpy as np
from numpy import arrange

import our_module as om
```

# Working with repositories

1. [Contribution flow](#contribution-flow)
1. [Git branching workflow](#git-branching-flow)
1. [Commit message format](#commit-message-format)
1. [Signing commits](#signing-commits)
1. [Creating pull requests](#pull-requests)
1. [Pull request review process](#pull-request-review-process)

## Contribution flow

1. Create a topic branch from where you want to base your work (always `main`).
1. Make commits of logical units.
1. Make sure your commit messages are in the proper format (see [Commit message format](#commit-message-format))
1. Push your changes to a topic branch.
1. Submit a pull request to the `main` branch.
1. Make sure the tests pass, and add any new tests as appropriate.

## Git branching workflow

We follow a semi-strict convention for branch workflow, commonly known as [GitHub Flow](https://guides.github.com/introduction/flow/). Feature branches are used to develop new features for the upcoming releases. Must branch off from `main` and must merge into `main`.

Branch name shall be descriptive.

If commits will be referring to the issue, the issue number / tag of the issue shall be included in in the branch name.

```text
fix-123-request-race-prevention
```

If commits will be referring to the task on YouTrack, they should include task prefix in the name.

```text
PBL-69-documentation-update
```

## Commit message format

We follow a strict convention for commit messages, known as [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) that is designed to answer two questions: what changed and why. The subject line should feature the what and the body of the commit should describe the why and how.

```text
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Fixes: #123
```

The format can be described more formally as follows:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

The first line is the subject and should be no longer than 70 characters, the second line is always blank, and other lines should be wrapped at 80 characters. This allows the message to be easier to read on GitHub as well as in various git tools.
