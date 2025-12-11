# PAINDGPR

This work presents an approach for interpreting 3D Ground Penetrating Radar (GPR) data by integrating machine learning techniques with classical signal processing.  
The objective is to detect hyperbolic structures within radargrams, reconstruct them in a three-dimensional context, and derive physically meaningful parameters.

This project was developed as part of my studies to become a Digital Engineer.  
The outcome is an ML-based solution that provides two complete workflows:

- one workflow for **detection**  
- one workflow for **training and evaluation**

---

# Repository Structure

<img width="352" height="445" alt="Repository Structure" src="https://github.com/user-attachments/assets/a4b429d8-88d2-4af2-b930-da5cfbb9ddf3" />

- **Annotated_Data** → Contains the annotated datasets used for training the models  
- **Data** → Includes subfolders containing fitting results, detection outputs, and more  
- **Notebooks** → Contains the Jupyter notebooks used to develop and document the full pipeline  
- **Model folders** → Contain the trained YOLO models

---

# How to Use the Pipeline

Two Jupyter notebooks guide you through the complete pipeline:

### Detection Workflow
Use the notebook:

**`8_Prediction_Workflow.ipynb`**

### Training Workflow
Use the notebook:

**`7_Training_and_Validating_Workflow.ipynb`**

Both notebooks describe the required steps to run detection, training, validation, and hyperbola fitting.

---

# Installation

To run the pipeline, you need to set up the Python environment using Conda.  
An `environment.yml` file is included in the repository and contains all required dependencies.

### 1. Clone the repository
```bash
git clone https://github.com/jogi96/PAINDGPR.git
cd PAINDGPR
