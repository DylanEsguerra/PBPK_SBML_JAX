# ARIA-E Models

This directory contains models related to amyloid-related imaging abnormalities with edema (ARIA-E), a significant side effect observed in Alzheimer's disease treatments using anti-amyloid antibodies.

## Project Overview

### Shared PK Module (`pk_module`)
- Provides a pharmacokinetic model used across different ARIA-E models
- Facilitates modularization and reuse of code

### Aldea2022 Model
- **Description**: Replication of the Aldea2022 model with vascular wall disturbance (VWD)
- **Key Features**:
  - SBML model definition
  - JAX-based numerical computation
  - VWD sub-module for detailed vascular modeling
- **Key Mechanisms**:
  - Antibody-mediated amyloid clearance
  - Vascular wall damage and repair 

### Extended Aldea2022 Model
- **Description**: Extends the Aldea2022 model by incorporating amyloid-beta production using Michaelis-Menten kinetics
- **Key Features**:
  - Dynamic amyloid-beta production
  - Enhanced model complexity

### Retout2022 Model
- **Description**: Models ARIA-E with a hazard function
- **Key Features**:
  - Utilizes the shared PK module
  - Incorporates hazard function for risk assessment

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/DylanEsguerra/PBPK_SBML_JAX.git
cd PBPK_SBML_JAX
```

2. Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your virtual environment is activated:
```bash
# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate
```

2. Run the models:
```bash
# For Aldea2022 model
python Aldea2022/src/run_Aldea_model.py

# For Extended Aldea2022 model
python Aldea2022/src/run_extended_model.py

# For Retout2022 model
python Retout2022/src/run_Retout_model.py
```

## File Structure and Functions

### Aldea2022 Model Files
- `src/run_Aldea_model.py`: Runs the Aldea2022 model
- `src/models/Aldea_modular_SBML.py`: SBML model generator for Aldea2022

### Extended Aldea2022 Model Files
- `src/run_extended_model.py`: Runs the extended Aldea2022 model
- `src/models/Aldea_extended_SBML.py`: SBML model generator for the extended model

### Retout2022 Model Files
- `src/run_Retout_model.py`: Runs the Retout2022 model
- `src/models/Retout_SBML.py`: SBML model generator for Retout2022

## Dependencies
- JAX: Automatic differentiation and numerical computing
- Diffrax: ODE solver
- libSBML: SBML model handling
- SBMLtoODEjax: SBML to JAX conversion
- Matplotlib: Result visualization

## References
Aldea, Roxana, et al. “In silico exploration of amyloid‐related imaging abnormalities in the gantenerumab open‐label extension trials using a semi‐mechanistic model.” Alzheimer’s &amp; Dementia: Translational Research &amp; Clinical Interventions, vol. 8, no. 1, Jan. 2022, https://doi.org/10.1002/trc2.12306.

Retout S, Gieschke R, Serafin D, Weber C, Frey N, Hofmann C. Disease modeling and model-based meta-analyses to define a new direction for a phase iii program of gantenerumab in alzheimer’s disease. Clin Pharmacol Ther. 2022;111(4):857-866. https://doi.org/10.1002/ cpt.2535