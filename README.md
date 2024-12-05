# PBPK Models Repository

This repository contains implementations of pharmacokinetic and pharmacodynamic models relevant to Alzheimer's disease research, including a detailed physiologically based pharmacokinetic model (PBPK) as well as simpler models with a specific focus on amyloid-related imaging abnormalities (ARIA). The models are implemented using Systems Biology Markup Language (SBML) for biological pathway definition and JAX for efficient numerical computations.

## Project Philosophy: Modular Model Development

The ultimate goal of this repository is to create a highly modularized framework for modeling various aspects of Alzheimer's disease. This approach offers several key advantages:

1. **Code Reuse**: Many published models share common components. For example, both the Aldea2022 and Retout2022 models use identical pharmacokinetic equations for antibody distribution. Our `pk_module` implements this shared functionality once and is used by both models.

2. **Extensibility**: New features can be added without modifying existing code. The extended Aldea model demonstrates this by adding Michaelis-Menten kinetics for amyloid-beta production while preserving the original model's vascular wall disturbance calculations.

3. **Model Integration**: As new Alzheimer's disease models are published each year, many share underlying mechanisms. Our modular approach allows researchers to:
   - Reuse validated components
   - Compare different mechanistic hypotheses
   - Combine features from multiple models

This modular philosophy is demonstrated throughout the repository:
- The `pk_module` is shared across multiple ARIA-E models
- The Aldea2022 model is extended without modifying the original implementation
- The Chang2019 PBPK model demonstrates a complicated multi-compartment model that is implemented in a modular fashion


## Biological Context

The models in this repository address two key areas in Alzheimer's disease research:

1. **Antibody Distribution**: How therapeutic antibodies move through the body and reach the brain
2. **ARIA-E Development**: How anti-amyloid treatments might lead to ARIA-E (edema) as a side effect


## Available Models

### [Chang et al. 2019](models/chang2019/)
- **Description**: PBPK model for antibody disposition in the brain
- **Key Features**:
  - SBML model definition
  - JAX-based numerical computation
  - Both full and modular implementations
- **Reference**: Chang HY, Wu S, Meno-Tetang G, Shah DK. A translational platform PBPK model for antibody disposition in the brain. J Pharmacokinet Pharmacodyn. 2019

### [ARIA-E Models](models/ARIA-E/)
- **Description**: Models related to ARIA-E, including replication and extended versions
- **Key Features**:
  - Shared PK model in `pk_module`
  - Replication of Aldea2022 with vascular wall disturbance (VWD)
  - Extended model with amyloid-beta production using Michaelis-Menten kinetics
  - Retout2022 model with hazard function
- **References**: 

Aldea, Roxana, et al. “In silico exploration of amyloid‐related imaging abnormalities in the gantenerumab open‐label extension trials using a semi‐mechanistic model.” Alzheimer’s &amp; Dementia: Translational Research &amp; Clinical Interventions, vol. 8, no. 1, Jan. 2022, https://doi.org/10.1002/trc2.12306.

Retout S, Gieschke R, Serafin D, Weber C, Frey N, Hofmann C. Disease modeling and model-based meta-analyses to define a new direction for a phase iii program of gantenerumab in alzheimer’s disease. Clin Pharmacol Ther. 2022;111(4):857-866. https://doi.org/10.1002/ cpt.2535

## Repository Structure
```
.
├── README.md             # This file
├── requirements.txt      # Project dependencies
├── utils/
│   └── printmath.py      # Utility for testing SBML compilation and viewing reactions
└── models/
    ├── chang2019/        # Chang et al. 2019 PBPK model
    │   ├── src/          # Model implementation
    │   ├── parameters/   # Model parameters
    │   ├── artifacts/    # Development artifacts and tests
    │   ├── generated/    # Generated code and figures
    │   └── README.md     # Model-specific documentation
    ├── ARIA-E/           # ARIA-E related models
    │   ├── pk_module/    # Shared PK model
    │   ├── Aldea2022/    # Aldea2022 model and its extensions
    │   │   ├── src/      # Model implementation
    │   │   ├── parameters/   # Model parameters
    │   │   ├── generated/    # Generated code and figures
    │   ├── Retout2022/   # Retout2022 model with hazard function
    │   │   ├── src/      # Model implementation
    │   │   ├── parameters/   # Model parameters
    │   │   ├── generated/    # Generated code and figures
    │   └── README.md     # Model-specific documentation
    └── common/           # Shared utilities
```

## Setup

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

## Testing SBML Models

The `utils/printmath.py` utility can be used to test SBML model compilation:

```bash
python utils/printmath.py path/to/your/model.xml
```

This will verify that the SBML file can be properly parsed and compiled.


