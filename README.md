# Project Description

A library for predicting customer churn. Implementation in Python.

# File and Data Description

## Tree Diagram

```
ROOT ${PROJECT_ROOT}
.... ChurnRate (module)
    |.... churn_library.py: (library)
    |.... constants.py: (configs)
.... data
.... images (for eda and results)
.... logs
.... models (for model weights and checkpoints)
.... tests (unit tests)
    |.... test_churn_library.py (unit tests for churn_library)
```

## Data and Models

Download model weights (optional) and data (required) from [gdrive]([placeholder](https://drive.google.com/drive/folders/1E38gtS1Ldq8L9MPigqI00D_05c9t4hux?usp=share_link)). Place models in ```models/``` and data in ```data/```

# Requirements

```pip install -r requiremnets.txt```

# Usage

## Example usage

from ${PROJECT_ROOT} run:
```python ChurnRate/churn_library.py```

## Running unit tests

from ${PROJECT_ROOT} run:
```pytest --disable-warnings```
**NOTE**: Some may will fail if as they depend on generated data from ```churn_library```