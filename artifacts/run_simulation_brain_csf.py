"""
Test case for circular dependencies between Brain and CSF.
When using this change parameters/pbpk_parameters.csv to 
set C_p_0 to 0.0 and C_p_lung_0 to original plasma value.
Simplified simulation runner for just the Brain and CSF modules.
This version uses the same parameter passing approach as run_simulation_b.py
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import diffrax
from diffrax import Tsit5, ODETerm
import matplotlib.pyplot as plt
import importlib
from pathlib import Path
import sys
import pandas as pd

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from src.main_b import main

def setup_coupled_system(registry, models):
    # Define state indices
    brain_indices = slice(0, 5)      # [C_p_brain, C_BBB_unbound_brain, C_BBB_bound_brain, C_is_brain, C_bc_brain]
    csf_indices = slice(5, 11)       # [C_BCSFB_unbound_brain, C_BCSFB_bound_brain, C_LV_brain, C_TFV_brain, C_CM_brain, C_SAS_brain]
    
    @jit
    def coupled_system(t, y, args):
        # Split states
        brain_states = y[brain_indices]
        csf_states = y[csf_indices]
        
        # Get parameters
        brain_params = models['brain'].c
        csf_params = models['csf'].c

        # Brain -> CSF coupling
        csf_params = csf_params.at[models['csf'].c_indexes['C_p_brain']].set(brain_states[models['brain'].y_indexes['C_p_brain']])
        csf_params = csf_params.at[models['csf'].c_indexes['C_is_brain']].set(brain_states[models['brain'].y_indexes['C_is_brain']])

        # CSF -> Brain coupling
        #brain_params = brain_params.at[models['brain'].c_indexes['C_SAS_brain']].set(csf_states[models['csf'].y_indexes['C_SAS_brain']])
        #brain_params = brain_params.at[models['brain'].c_indexes['C_BCSFB_bound_brain']].set(csf_states[models['csf'].y_indexes['C_BCSFB_bound_brain']])
        
        # Calculate derivatives using current parameters
        dy_brain_dt = models['brain'].RateofSpeciesChange()(brain_states, t, {}, brain_params)
        dy_csf_dt = models['csf'].RateofSpeciesChange()(csf_states, t, {}, csf_params)

        # Combine derivatives
        dy_dt = jnp.concatenate([dy_brain_dt, dy_csf_dt])
        
        return dy_dt

    return coupled_system

def run_simulation():
    # Initialize registry and process modules
    registry = main()
    
    # Load parameters from CSV
    params_df = pd.read_csv('parameters/pbpk_parameters.csv')
    
    # Get models in fixed order
    models = {}
    for module_name in ['brain', 'csf']:
        model_module = importlib.import_module(f"generated.jax.{module_name}_jax")
        importlib.reload(model_module)
        models[module_name] = model_module
    
    # Create coupled system
    coupled_system = setup_coupled_system(registry, models)
    
    # Get initial conditions in correct order
    y0_combined = jnp.concatenate([
        models['brain'].y0,
        models['csf'].y0
    ])
    
    # Simulation parameters
    t0 = 0.0
    t1 = 2000
    dt = 0.001
    n_steps = 2000
    
    # Create diffrax solver
    term = diffrax.ODETerm(coupled_system)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))

    # Solve the ODE system
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0_combined,
        args=(None,),
        saveat=saveat,
        max_steps=1000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8)
    )
    
    return sol, registry

def plot_results(sol, registry):
    # Create figure with 1x2 grid layout for compartment plots
    fig1 = plt.figure(figsize=(15, 6))
    gs = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Brain compartments (indices 0-4)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.semilogy(sol.ts, sol.ys[:, 0], label='Brain Plasma', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 1], label='BBB Unbound', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 2], label='BBB Bound', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 3], label='Brain ISF', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 4], label='Brain Blood Cells', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Brain Compartments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CSF compartments (indices 5-10)
    ax2 = fig1.add_subplot(gs[0, 1])
    ax2.semilogy(sol.ts, sol.ys[:, 5], label='BCSFB Unbound', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 6], label='BCSFB Bound', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 7], label='Lateral Ventricle', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 8], label='Third/Fourth Ventricle', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 9], label='Cisterna Magna', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 10], label='Subarachnoid Space', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('CSF Compartments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Save figure
    fig1.savefig('brain_csf_concentration_plots.png', dpi=300, bbox_inches='tight')
    
    plt.show()

    # Print final concentrations
    print("\nFinal Concentrations:")
    print("-" * 30)
    compartments = [
        'Brain:', 
        '  Plasma', '  BBB Unbound', '  BBB Bound',
        '  ISF', '  Blood Cells',
        'CSF:',
        '  BCSFB Unbound', '  BCSFB Bound', '  Lateral Ventricle', 
        '  Third/Fourth Ventricle', '  Cisterna Magna', '  Subarachnoid Space'
    ]

    for i, comp in enumerate(compartments):
        if comp.endswith(':'):
            print(f"\n{comp}")
        else:
            print(f"{comp:<25}: {sol.ys[-1, i-1]:.6f}")

if __name__ == "__main__":
    sol, registry = run_simulation()
    plot_results(sol, registry) 