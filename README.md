# 3DAnnotatePropWizard

## Overview
**3DAnnotatePropWizard** is a smart labeling tool designed to help you efficiently annotate 3D volumes. Leveraging the Segment Anything Model (SAM), this tool propagates 2D annotations through 3D volumes, significantly speeding up the labeling process and enhancing accuracy.

## Features
- **Smart Annotation Propagation:** Uses 2D SAM to propagate annotations across 3D volumes.
- **User-Friendly Interface:** Intuitive UI designed for ease of use.
- **High Accuracy:** Ensures precise annotations through advanced algorithms.
- **Fast Processing:** Accelerates the labeling process with intelligent propagation techniques.

## Installation
To get started with 3DAnnotatePropWizard, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/3DLabelWizard.git
    cd 3DLabelWizard
    ```
2. **Install dependencies:**

    [TBD]
    ```sh
    poetry install --with dev -E cutie --sync
    ```

## Usage
Load your 3D volume data into the application.
Annotate a slice using the 2D annotation tools provided.
Propagate annotations through the 3D volume using the propagation feature.
Review and refine the propagated annotations as needed.

### MaskPropagation
```python
from annotatepropwizard3d.mask_propagation import MaskPropagation

model = MaskPropagation("{sam weight path}", "{cutie yaml path}")

# 3d numpy array from nii
# 3d numpy array from labeled mask
with torch.inference_mode():
    result = model.predict_by_volume({3d numpy array}, {3d numpy array})

# result is a 3d numpy array
```

### XYrollPrediction
```python
from annotatepropwizard3d.xyroll_prediction import XYrollPrediction

model = XYrollPrediction("{cutie yaml path}")

# 3d numpy array from nii
# 2d numpy array from target labeled slice
# int k for target slice index
with torch.inference_mode():
    result = model.predict({3d numpy array}, {2d numpy array}, {int k}})

# result is a 3d numpy array
```