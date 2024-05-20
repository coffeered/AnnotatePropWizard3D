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
    poetry install
    ```

## Usage
Load your 3D volume data into the application.
Annotate a slice using the 2D annotation tools provided.
Propagate annotations through the 3D volume using the propagation feature.
Review and refine the propagated annotations as needed.
