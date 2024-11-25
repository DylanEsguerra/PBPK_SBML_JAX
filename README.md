# PBPK Model Implementation (Chang et al. 2019)

This repository implements a Physiologically Based Pharmacokinetic (PBPK) model for antibody disposition in the brain, based on Chang et al. 2019. The implementation uses SBML for model definition, SBMLtoODEjax for conversion to JAX code, and Diffrax for solving the differential equations.

## Project Overview

The project offers two implementation approaches:
1. A unified PBPK model (core implementation)
2. A modular organ-based approach (modular implementation)

### Core Implementation Workflow
1. Parameter loading (`parameters/pbpk_parameters.csv`)
2. SBML model generation (`src/models/PBPK_SBML.py`)
3. JAX code generation (using SBMLtoODEjax)
4. ODE solving with Diffrax (`src/run_PBPK.py`)

### Modular Implementation Workflow
1. Module registration and dependency management
2. Per-organ parameter loading (`parameters/*_params.xlsx`)
3. Individual SBML model generation for each organ
4. Coupled system solving

## File Structure and Functions

### Core Files
- `src/run_PBPK.py`: Main entry point
  - Generates JAX model from SBML
  - Sets up and runs the ODE solver
  - Handles result plotting

- `src/models/PBPK_SBML.py`: SBML model generator
  - Creates compartments for all organs
  - Defines species and their initial conditions
  - Implements all ODEs and rate rules
  - Validates model structure

### Modular Implementation Files

- `src/run_simulation.py`: Modular solver with explicit dependencies
  - manually defines relationships between modules
  - best to use when modifying the model structure and attempting to resolve circular dependency issue 
- `src/run_simulation_2.py`: Modular solver with registry
  - Uses ModuleRegistry for organ management
  - Handles module dependencies and coupling
  - Coordinates the multi-organ simulation

- `src/models/[organ]/*`: Organ-specific modules
  - `[organ]_sbml.py`: SBML model generator
  - `[organ]_solver.py`: Solver configuration
  - Parameters in `parameters/[organ]_params.xlsx`

Current modules:
- Blood: Central circulation system
- Brain: Blood-brain barrier and brain tissue
- CSF: Cerebrospinal fluid dynamics
- Lung: Pulmonary circulation
- Liver: Hepatic processing

## Parameter Organization
- Global parameters: `parameters/pbpk_parameters.csv`
- Organ-specific parameters: `parameters/[organ]_params.xlsx`
  - Volumes
  - Flows
  - Kinetics
  - Concentrations

## Generated Files
- `generated/sbml/`: SBML model definitions
  - `pbpk_model.xml`: Unified model
  - `[organ]_sbml.xml`: Organ-specific models
- `generated/jax/`: Generated JAX code
  - Used by solvers for numerical integration

  ## Usage

### Running the Full Model
```bash
python src/run_PBPK.py
```

### Running the Modular Implementation
```bash
python src/run_simulation.py
```

## Dependencies
- JAX: Automatic differentiation and numerical computing
- Diffrax: ODE solver
- libSBML: SBML model handling
- SBMLtoODEjax: SBML to JAX conversion
- Matplotlib: Result visualization
- Pandas: Parameter data handling

## Model Details
The PBPK model includes:
- Compartmental structure for each organ
- Blood flow and lymphatic circulation
- FcRn-mediated transport
- Blood-brain barrier dynamics
- CSF flow and clearance

## References
Chang HY, Wu S, Meno-Tetang G, Shah DK. A translational platform PBPK model for antibody disposition in the brain. J Pharmacokinet Pharmacodyn. 2019 Aug;46(4):319-338.

## Implementation Challenges and Current Status

### Core vs Modular Implementation
The core implementation (`run_PBPK.py`) successfully solves the complete PBPK model by treating all compartments as part of a single system. This approach works well because all concentrations are updated simultaneously in each timestep.

### Circular Dependencies in Modular Version
The modular implementation (`run_simulation.py`) faces challenges with circular dependencies due to bidirectional flows:

1. **Blood-Tissue Cycle**:
   - Blood → Lung → All Tissues
   - All Tissues → Blood (return flow)

2. **Brain-CSF Cycle**:
   - Brain → CSF (ISF to CSF spaces)
   - CSF → Brain (SAS to brain tissue)

#### Current Workaround
In `run_simulation.py`, these circular flows can be controlled by commenting/uncommenting the return flow coupling sections:

```python
# Return flow coupling (commented out by default)
#blood_params = blood_params.at[models['blood'].c_indexes['C_p_brain']].set(brain_states[models['brain'].y_indexes['C_p_brain']])
#blood_params = blood_params.at[models['blood'].c_indexes['C_bc_brain']].set(brain_states[models['brain'].y_indexes['C_bc_brain']])
# ... additional return flows ...
```

- **With Return Flows Disabled**: The simulation shows logical forward flow from blood → lung → tissues
- **With Return Flows Enabled**: The solution contains NaN values due to unresolved circular dependencies

### Ongoing Development
Work is continuing to resolve these circular dependencies while maintaining the modular structure. Potential approaches include:
- Implementing a sequential update scheme
- Using previous timestep values for return flows
- Developing a hybrid approach that preserves modularity while handling circular dependencies

For now, users can explore the system behavior using `run_simulation.py` with return flows disabled, or use the complete solution in `run_PBPK.py`.



