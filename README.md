# Description

A library for predicting customer churn. Implementation in Python.

# Requirements

```pip install -r requiremnets.txt```
Download model weights (optional) and data (required) from [gdrive]([placeholder](https://drive.google.com/drive/folders/1E38gtS1Ldq8L9MPigqI00D_05c9t4hux?usp=share_link)). Place models in ```models/``` and data in ```data/```

# Usage

## Running end2end script

```python ChurnRate/churn_library.py```

## Running unit tests

```pytest --disable-warnings``` from project root
**NOTE**: Some may will fail if as they depend on generated data from ```churn_library```