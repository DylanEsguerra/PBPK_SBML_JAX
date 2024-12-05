"""
Main module for the PBPK simulation system.
This module handles:
- Registry setup and configuration
- Parameter loading from CSV
- SBML model generation
- JAX code generation from SBML
- Module dependency management
"""
import jax
jax.config.update("jax_enable_x64", True)
import pandas as pd
from pathlib import Path
import importlib

from models.blood import blood_sbml
from models.brain import brain_sbml
from models.csf import csf_sbml
from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse
from registry_b import ModuleRegistry, ModuleConfig

def load_all_params():
    """Load all parameters from the consolidated CSV file"""
    params_df = pd.read_csv('parameters/pbpk_parameters.csv')
    return {row['name']: row['value'] for _, row in params_df.iterrows()}

def setup_registry():
    # Create output directories if they don't exist
    generated_dir = Path("generated")
    generated_sbml = generated_dir / "sbml"
    generated_jax = generated_dir / "jax"
    for dir in [generated_dir, generated_sbml, generated_jax]:
        dir.mkdir(parents=True, exist_ok=True)
    
    registry = ModuleRegistry()
    
    # Register modules with simplified configs
    registry.register_module(ModuleConfig(
        name="blood",
        dependencies=['brain', 'csf', 'lung', 'liver'],
        xml_path=Path("generated/sbml/blood_sbml.xml"),
        jax_path=Path("generated/jax/blood_jax.py"),
        initial_conditions=['C_p_0', 'C_bc_0', 'C_ln_0'],
        coupling_params={
            'C_p_brain': 'brain',
            'C_bc_brain': 'brain',
            'C_is_brain': 'brain',
            'C_SAS_brain': 'csf',
            'C_is_lung': 'lung',
            'C_p_liver': 'liver',
            'C_bc_liver': 'liver',
            'C_is_liver': 'liver'
        }
    ))

    # Register Brain Module
    registry.register_module(ModuleConfig(
        name="brain",
        dependencies=['blood', 'lung', 'csf'],
        xml_path=Path("generated/sbml/brain_sbml.xml"),
        jax_path=Path("generated/jax/brain_jax.py"),
        initial_conditions=[
            'C_p_brain_0', 'C_bc_brain_0', 'C_BBB_unbound_brain_0',
            'C_BBB_bound_brain_0', 'C_is_brain_0'
        ],
        coupling_params={
            'C_p_lung': 'lung',
            'C_bc_lung': 'lung',
            'C_SAS_brain': 'csf',
            'C_BCSFB_bound_brain': 'csf',
            'C_BCSFB_unbound_brain': 'csf'
        }
    ))

    # Register CSF Module - Fix typo and add missing couplings
    registry.register_module(ModuleConfig(
        name="csf",
        dependencies=['brain', 'lung'],
        xml_path=Path("generated/sbml/csf_sbml.xml"),
        jax_path=Path("generated/jax/csf_jax.py"),
        initial_conditions=[
            'C_BCSFB_unbound_brain_0', 'C_BCSFB_bound_brain_0',
            'C_LV_brain_0', 'C_TFV_brain_0', 'C_CM_brain_0', 'C_SAS_brain_0'
        ],
        coupling_params={
            'C_p_brain': 'brain',
            'C_is_brain': 'brain',
        }
    ))

    # Register Lung Module - Add blood feedback
    registry.register_module(ModuleConfig(
        name="lung",
        dependencies=['blood'],
        xml_path=Path("generated/sbml/lung_sbml.xml"),
        jax_path=Path("generated/jax/lung_jax.py"),
        initial_conditions=[
            'C_p_lung_0', 'C_bc_lung_0', 'C_is_lung_0',
            'C_e_unbound_lung_0', 'C_e_bound_lung_0', 'FcRn_free_lung_0'
        ],
        coupling_params={
            'C_p': 'blood',
            'C_bc': 'blood',
            'C_is': 'blood'
        }
    ))

    # Register Liver Module - Add blood coupling
    registry.register_module(ModuleConfig(
        name="liver",
        dependencies=['blood', 'lung'],
        xml_path=Path("generated/sbml/liver_sbml.xml"),
        jax_path=Path("generated/jax/liver_jax.py"),
        initial_conditions=[
            'C_p_liver_0', 'C_bc_liver_0', 'C_is_liver_0',
            'C_e_unbound_liver_0', 'C_e_bound_liver_0', 'FcRn_free_liver_0'
        ],
        coupling_params={
            'C_p_lung': 'lung',
            'C_bc_lung': 'lung',
        }
    ))


    return registry

def main():
    """Main function to set up and run the PBPK model"""
    registry = setup_registry()
    all_params = load_all_params()
    
    # Process each module in order
    for module_name in registry.get_execution_order():
        config = registry.modules[module_name]
        print(f"\nProcessing {module_name} module...")
        
        # Generate SBML file using all parameters
        module = importlib.import_module(f"models.{module_name}.{module_name}_sbml")
        module.main(all_params)
        
        # Generate JAX model
        xml_path = Path("generated/sbml") / f"{module_name}_sbml.xml"
        model_data = parse.ParseSBMLFile(str(xml_path))
        GenerateModel(model_data, str(config.jax_path))

    return registry

if __name__ == "__main__":
    main() 