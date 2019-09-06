---
name: "\U0001F41EBug report"
about: Submit a bug report
title: ''
labels: bug
assignees: ''

---

## 🐞Describe the bug
A clear and brief description of what the bug is.

## Trace
If applicable, please paste the error trace.

## To Reproduce
- If a python script can reproduce the error, please paste the code snippet
```
from tfcoreml import convert
# Paste code snippet here
```
- If applicable, please attach tensorflow model
    - If model cannot be shared publicly, please attach it via filing a bug report at https://developer.apple.com/bug-reporting/ 
- If model conversion succeeds, however, there is numerical mismatch between the original and the coreml model, please paste python script used for comparison (tf code etc.)

## System environment (please complete the following information):
 - coremltools version  (e.g., 3.0b5):
 - tf-coreml version (e.g. 0.4.0b1)
 - tensorflow version (e.g. 1.14):
 - OS (e.g., MacOS, Linux):
 - macOS version (if applicable):
 - How you install python (anaconda, virtualenv, system):
 - python version (e.g. 3.7):
 - any other relevant information:

## Additional context
Add any other context about the problem here.
