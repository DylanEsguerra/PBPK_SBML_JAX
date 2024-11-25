# PBPK Model Implementation (Chang et al. 2019)

This repository contains a JAX-based implementation of the Physiologically Based Pharmacokinetic (PBPK) model described in Chang et al. 2019. The model is implemented using SBML (Systems Biology Markup Language) and converted to JAX code using SBMLtoODEjax, with differential equations solved using Diffrax.

## Project Structure

### Core Implementation
- `src/run_PBPK.py`: Main entry point for running the full PBPK model implementation
  - Calls `models/PBPK_SBML.py` to generate the SBML model
  - Uses SBMLtoODEjax to convert SBML to JAX code
  - Implements the solver using Diffrax

- `src/models/PBPK_SBML.py`: Creates the unified PBPK SBML model
  - Defines compartments, parameters, and equations
  - Handles parameter loading from CSV files
  - Generates and validates the SBML model

### Modular Implementation
- `src/run_simulation.py`: Alternative implementation using a modular approach
  - Implements organ-based modularization
  - Handles coupling between different organ modules
  - Uses Diffrax for solving the coupled system
  - Passes shared species concentrations between modules within ODE solver 

- `src/run_simulation_2.py`: Enhanced modular implementation
  - Uses `ModuleRegistry` for managing organ modules
  - Provides more flexible module addition and coupling
    

### Module Components
Each organ module follows a similar structure:
- SBML model creation (`*_sbml.py`)
- Parameter handling
- JAX code generation

Modules include:
- Blood
- Brain
- CSF (Cerebrospinal Fluid)
- Lung
- Liver

## Dependencies
- JAX
- Diffrax
- libSBML
- SBMLtoODEjax
- Matplotlib
- Pandas


## Usage

### Running the Full Model
```bash
python src/run_PBPK.py
```

### Running the Modular Implementation
```bash
python src/run_simulation.py
# or
python src/run_simulation_2.py
```

## Model Structure

### Full PBPK Model
The complete model integrates all organ systems into a single SBML model, including:
- Plasma and blood cell compartments
- Organ-specific compartments (brain, lung, liver, etc.)
- Lymphatic system
- Binding kinetics and transport mechanisms

### Modular Implementation
The modular version separates the model into individual organ modules that can be:
- Developed and tested independently
- Coupled through a registry system
- Extended with new organs/components
- Configured with separate parameter sets

## Parameter Files
- Located in `parameters/` directory
- Organ-specific parameters in Excel format 
- Global parameters for the full model stored in a CSV file

## Generated Files
- SBML models: `generated/sbml/`
- JAX code: `generated/jax/`

## References
Chang HY, Wu S, Meno-Tetang G, Shah DK. A translational platform PBPK model for antibody disposition in the brain. J Pharmacokinet Pharmacodyn. 2019 Aug;46(4):319-338. doi: 10.1007/s10928-019-09641-8. Epub 2019 May 21. PMID: 31115858; PMCID: PMC8409011.

