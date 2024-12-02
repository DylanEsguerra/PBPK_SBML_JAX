"""
Simulation runner for the combined Brain-CSF model.
This version uses the combined SBML model instead of coupling separate models.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import importlib
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.models.combined_brain_csf_model import create_combined_model, save_model

def run_simulation():
    # Load parameters from CSV
    params_df = pd.read_csv('parameters/pbpk_parameters.csv')
    params_dict = dict(zip(params_df['name'], params_df['value']))
    
    # Create and save combined model
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_brain_csf.xml"
    
    document = create_combined_model(params_dict)
    save_model(document, str(output_path))
    
    # Generate JAX code from combined model
    from sbmltoodejax.modulegeneration import GenerateModel
    from sbmltoodejax import parse
    
    jax_dir = Path("generated/jax")
    jax_dir.mkdir(parents=True, exist_ok=True)
    jax_path = jax_dir / "combined_brain_csf_jax.py"
    
    model_data = parse.ParseSBMLFile(str(output_path))
    GenerateModel(model_data, str(jax_path))
    
    # Import generated JAX module
    combined_module = importlib.import_module("generated.jax.combined_brain_csf_jax")
    importlib.reload(combined_module)
    
    # Simulation parameters
    t0 = 0.0
    t1 = 2000
    dt = 0.001
    n_steps = 2000
    
    # Create diffrax solver
    term = diffrax.ODETerm(lambda t, y, args: combined_module.RateofSpeciesChange()(y, t, args[0], args[1]))
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))

    # Solve the ODE system
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=combined_module.y0,
        args=(None, combined_module.c),
        saveat=saveat,
        max_steps=1000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8)
    )
    
    return sol, combined_module

def plot_results(sol, module):
    # Create figure with 1x2 grid layout for compartment plots
    fig1 = plt.figure(figsize=(15, 6))
    gs = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Brain compartments
    ax1 = fig1.add_subplot(gs[0, 0])
    brain_vars = ['C_p_brain', 'C_BBB_unbound_brain', 'C_BBB_bound_brain', 
                 'C_is_brain', 'C_bc_brain']
    for var in brain_vars:
        idx = module.y_indexes[var]
        ax1.semilogy(sol.ts, sol.ys[:, idx], label=var.replace('_brain', ''), linewidth=2)
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Brain Compartments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CSF compartments
    ax2 = fig1.add_subplot(gs[0, 1])
    csf_vars = ['C_BCSFB_unbound_brain', 'C_BCSFB_bound_brain', 'C_LV_brain',
                'C_TFV_brain', 'C_CM_brain', 'C_SAS_brain']
    for var in csf_vars:
        idx = module.y_indexes[var]
        ax2.semilogy(sol.ts, sol.ys[:, idx], label=var.replace('_brain', ''), linewidth=2)
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('CSF Compartments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Save figure
    fig1.savefig('combined_brain_csf_concentration_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print final concentrations
    print("\nFinal Concentrations:")
    print("-" * 30)
    
    print("\nBrain:")
    for var in brain_vars:
        idx = module.y_indexes[var]
        print(f"  {var:<25}: {sol.ys[-1, idx]:.6f}")
    
    print("\nCSF:")
    for var in csf_vars:
        idx = module.y_indexes[var]
        print(f"  {var:<25}: {sol.ys[-1, idx]:.6f}")

if __name__ == "__main__":
    sol, module = run_simulation()
    plot_results(sol, module) 