"""
Simulation runner for the combined Brain-CSF system using the master model approach.
This version uses SBML's hierarchical model composition to create a single unified model,
then converts it to JAX using sbmltoodejax.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import diffrax
from diffrax import Tsit5, ODETerm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
import importlib
from sbmltoodejax.modulegeneration import GenerateModel 
from sbmltoodejax import parse
import libsbml

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from src.models.master_model import create_master_model


def run_simulation():
    # Load parameters from CSV
    params_df = pd.read_csv('parameters/pbpk_parameters.csv')

    # Convert DataFrame to dictionary for easier parameter access
    params_dict = dict(zip(params_df['name'], params_df['value']))
    
    # Create master model
    master_document = create_master_model(params_dict)
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "master_sbml.xml"
    
    libsbml.writeSBMLToFile(master_document, str(master_path))
    print(f"Master model saved to {master_path}")
    
    # Generate JAX model from master SBML
    jax_dir = Path("generated/jax")
    jax_dir.mkdir(parents=True, exist_ok=True)
    jax_path = jax_dir / "master_jax.py"
    
    model_data = parse.ParseSBMLFile(str(master_path))
    GenerateModel(model_data, str(jax_path))
    
    # Import generated JAX model
    master_module = importlib.import_module("generated.jax.master_jax")
    importlib.reload(master_module)
    
    # Set up simulation parameters
    t0 = 0.0
    t1 = 2000
    dt = 0.001
    n_steps = 2000
    
    # Create solver
    term = ODETerm(lambda t, y, args: master_module.RateofSpeciesChange()(y, t, args[0], args[1]))
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))
    
    # Solve system using initial conditions from the master model
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=master_module.y0,
        args=(None, master_module.c),
        saveat=saveat,
        max_steps=1000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8)
    )
    
    return sol, master_module

def plot_results(sol, master_module):
    # Create figure with subplots for each organ system
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Define species groups for each organ
    organ_species = {
        'Blood': ['C_p', 'C_bc', 'C_ln'],
        'Lung': ['C_p_lung', 'C_bc_lung', 'C_is_lung', 'C_e_unbound_lung', 'C_e_bound_lung'],
        'Brain': ['C_p_brain', 'C_BBB_unbound_brain', 'C_BBB_bound_brain', 'C_is_brain', 'C_bc_brain'],
        'CSF': ['C_BCSFB_unbound_brain', 'C_BCSFB_bound_brain', 'C_LV_brain', 'C_TFV_brain', 'C_CM_brain', 'C_SAS_brain'],
        'Liver': ['C_p_liver', 'C_bc_liver', 'C_is_liver', 'C_e_unbound_liver', 'C_e_bound_liver']
    }
    
    # Plot each organ system
    for idx, (organ, species_list) in enumerate(organ_species.items()):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Get indices for each species
        indices = [master_module.y_indexes[species] for species in species_list]
        
        # Plot each species
        for idx, label in zip(indices, species_list):
            ax.semilogy(sol.ts, sol.ys[:, idx], label=label, linewidth=2)
            
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{organ} Compartments')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig('all_organs_master_model_concentration_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final concentrations
    print("\nFinal Concentrations:")
    print("-" * 30)
    for organ, species_list in organ_species.items():
        print(f"\n{organ}:")
        for species in species_list:
            idx = master_module.y_indexes[species]
            print(f"  {species:<25}: {sol.ys[-1, idx]:.6f}")

if __name__ == "__main__":
    sol, master_module = run_simulation()
    plot_results(sol, master_module) 