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
    master_document = create_master_model(params_dict, params_dict)  # Same params for both brain and CSF
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
    importlib.reload(master_module)  # Ensure we have the latest version
    
    # Simulation parameters
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
        args=(None, master_module.c),  # Pass parameters from generated model
        saveat=saveat,
        max_steps=1000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8)
    )
    
    return sol, master_module

def plot_results(sol, master_module):
    # Create figure with 1x2 grid layout for compartment plots
    fig1 = plt.figure(figsize=(15, 6))
    gs = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Get indices from the master module
    brain_species = ['C_p_brain', 'C_BBB_unbound_brain', 'C_BBB_bound_brain', 
                    'C_is_brain', 'C_bc_brain']
    csf_species = ['C_BCSFB_unbound_brain', 'C_BCSFB_bound_brain', 'C_LV_brain',
                  'C_TFV_brain', 'C_CM_brain', 'C_SAS_brain']
    
    # Map species names to indices
    brain_indices = [master_module.y_indexes[species] for species in brain_species]
    csf_indices = [master_module.y_indexes[species] for species in csf_species]
    
    # Brain compartments plot
    ax1 = fig1.add_subplot(gs[0, 0])
    for idx, label in zip(brain_indices, brain_species):
        ax1.semilogy(sol.ts, sol.ys[:, idx], label=label.replace('_brain', ''), linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Brain Compartments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CSF compartments plot
    ax2 = fig1.add_subplot(gs[0, 1])
    for idx, label in zip(csf_indices, csf_species):
        ax2.semilogy(sol.ts, sol.ys[:, idx], label=label.replace('_brain', ''), linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('CSF Compartments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Save figure
    fig1.savefig('brain_csf_master_concentration_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print final concentrations
    print("\nFinal Concentrations:")
    print("-" * 30)
    all_species = [('Brain:', brain_species), ('CSF:', csf_species)]
    
    for header, species_list in all_species:
        print(f"\n{header}")
        for species in species_list:
            idx = master_module.y_indexes[species]
            print(f"  {species:<25}: {sol.ys[-1, idx]:.6f}")

if __name__ == "__main__":
    sol, master_module = run_simulation()
    plot_results(sol, master_module) 