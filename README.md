# PBPK Models Repository

This repository contains implementations of various Physiologically Based Pharmacokinetic (PBPK) models using SBML for model definition and JAX for numerical computation.

## Available Models

### [Chang et al. 2019](models/chang2019/)
- **Description**: PBPK model for antibody disposition in the brain
- **Key Features**:
  - SBML model definition
  - JAX-based numerical computation
  - Both full and modular implementations
- **Reference**: Chang HY, Wu S, Meno-Tetang G, Shah DK. A translational platform PBPK model for antibody disposition in the brain. J Pharmacokinet Pharmacodyn. 2019

## Repository Structure
```
.
├── README.md             # This file
├── requirements.txt       # Project dependencies
├── utils/
│   └── printmath.py      # Utility for testing SBML compilation
└── models/
    ├── chang2019/        # Chang et al. 2019 PBPK model
    │   ├── src/          # Model implementation
    │   ├── parameters/   # Model parameters
    │   ├── artifacts/    # Development artifacts and tests
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

## Contributing

If you'd like to add a new PBPK model to this repository:
1. Create a new directory under `models/`
2. Include model-specific documentation in a README.md
3. Follow the existing structure for consistency
4. Use `utils/printmath.py` to verify your SBML models

## License

[Your license information]
