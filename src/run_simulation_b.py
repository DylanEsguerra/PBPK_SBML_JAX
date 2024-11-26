"""
Simulation runner for the PBPK model using a modular approach.
This file handles the setup of coupled systems, running simulations, and plotting results
for individual organ modules (blood, lung, brain, CSF, liver). 

This version uses the consolidated parameter file (pbpk_parameters.csv).
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
from src.main_b import main, load_all_params

def setup_coupled_system(registry, models, all_params):
    # Define state indices
    blood_indices = slice(0, 3)      # [C_p, C_bc, C_ln]
    lung_indices = slice(3, 9)       # [C_p_lung, C_bc_lung, C_e_unbound_lung, C_e_bound_lung, C_is_lung, FcRn_free_lung]
    brain_indices = slice(9, 14)     # [C_p_brain, C_BBB_unbound_brain, C_BBB_bound_brain, C_is_brain, C_bc_brain]
    csf_indices = slice(14, 20)      # [C_BCSFB_unbound_brain, C_BCSFB_bound_brain, C_LV_brain, C_TFV_brain, C_CM_brain, C_SAS_brain]
    liver_indices = slice(20, 26)    # [C_p_liver, C_bc_liver, C_e_unbound_liver, C_e_bound_liver, C_is_liver, FcRn_free_liver]
    
    @jit
    def coupled_system(t, y, args):
        # Split states
        blood_states = y[blood_indices]
        lung_states = y[lung_indices]
        brain_states = y[brain_indices]
        csf_states = y[csf_indices]
        liver_states = y[liver_indices]
        
        # Get parameters
        blood_params = models['blood'].c
        lung_params = models['lung'].c
        brain_params = models['brain'].c
        csf_params = models['csf'].c
        liver_params = models['liver'].c

        # Forward flow coupling
        lung_params = lung_params.at[models['lung'].c_indexes['C_p']].set(blood_states[models['blood'].y_indexes['C_p']])
        lung_params = lung_params.at[models['lung'].c_indexes['C_bc']].set(blood_states[models['blood'].y_indexes['C_bc']])

        brain_params = brain_params.at[models['brain'].c_indexes['C_p_lung']].set(lung_states[models['lung'].y_indexes['C_p_lung']])
        brain_params = brain_params.at[models['brain'].c_indexes['C_bc_lung']].set(lung_states[models['lung'].y_indexes['C_bc_lung']])

        csf_params = csf_params.at[models['csf'].c_indexes['C_p_brain']].set(brain_states[models['brain'].y_indexes['C_p_brain']])
        csf_params = csf_params.at[models['csf'].c_indexes['C_is_brain']].set(brain_states[models['brain'].y_indexes['C_is_brain']])

        liver_params = liver_params.at[models['liver'].c_indexes['C_p_lung']].set(lung_states[models['lung'].y_indexes['C_p_lung']])
        liver_params = liver_params.at[models['liver'].c_indexes['C_bc_lung']].set(lung_states[models['lung'].y_indexes['C_bc_lung']])


        # Return flow coupling # comment out to disable
        #blood_params = blood_params.at[models['blood'].c_indexes['C_p_brain']].set(brain_states[models['brain'].y_indexes['C_p_brain']])
        #blood_params = blood_params.at[models['blood'].c_indexes['C_bc_brain']].set(brain_states[models['brain'].y_indexes['C_bc_brain']])
        #blood_params = blood_params.at[models['blood'].c_indexes['C_is_brain']].set(brain_states[models['brain'].y_indexes['C_is_brain']])
        #blood_params = blood_params.at[models['blood'].c_indexes['C_is_lung']].set(lung_states[models['lung'].y_indexes['C_is_lung']])
        #blood_params = blood_params.at[models['blood'].c_indexes['C_SAS_brain']].set(csf_states[models['csf'].y_indexes['C_SAS_brain']])
        #blood_params = blood_params.at[models['blood'].c_indexes['C_is_liver']].set(liver_states[models['liver'].y_indexes['C_is_liver']])

        #brain_params = brain_params.at[models['brain'].c_indexes['C_SAS_brain']].set(csf_states[models['csf'].y_indexes['C_SAS_brain']])
        #brain_params = brain_params.at[models['brain'].c_indexes['C_BCSFB_bound_brain']].set(csf_states[models['csf'].y_indexes['C_BCSFB_bound_brain']])
        
         # Calculate derivatives using current parameters
        dy_blood_dt = models['blood'].RateofSpeciesChange()(blood_states, t, {}, blood_params)
        dy_lung_dt = models['lung'].RateofSpeciesChange()(lung_states, t, {}, lung_params)
        dy_brain_dt = models['brain'].RateofSpeciesChange()(brain_states, t, {}, brain_params)
        dy_csf_dt = models['csf'].RateofSpeciesChange()(csf_states, t, {}, csf_params)
        dy_liver_dt = models['liver'].RateofSpeciesChange()(liver_states, t, {}, liver_params)

        # Combine derivatives
        dy_dt = jnp.concatenate([dy_blood_dt, dy_lung_dt, dy_brain_dt, dy_csf_dt, dy_liver_dt])
        #dy_dt = jnp.nan_to_num(dy_dt, nan=0.0)

        return dy_dt

    return coupled_system

def run_simulation():
    # Initialize registry and process modules
    registry = main()
    
    # Load parameters from CSV
    params_df = pd.read_csv('parameters/pbpk_parameters.csv')
    
    # Get models in fixed order
    models = {}
    for module_name in ['blood', 'lung', 'brain', 'csf', 'liver']:
        model_module = importlib.import_module(f"generated.jax.{module_name}_jax")
        importlib.reload(model_module)
        models[module_name] = model_module
    
    # Create coupled system
    coupled_system = setup_coupled_system(registry, models, params_df)
    
    # Get initial conditions in correct order
    y0_combined = jnp.concatenate([
        models['blood'].y0,
        models['lung'].y0,
        models['brain'].y0,
        models['csf'].y0,
        models['liver'].y0
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
    # Create figure with 2x3 grid layout for compartment plots
    fig1 = plt.figure(figsize=(15, 10))
    gs = fig1.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Blood compartments (indices 0-2)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.semilogy(sol.ts, sol.ys[:, 0], label='Blood Plasma', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 1], label='Blood Cells', linewidth=2)
    ax1.semilogy(sol.ts, sol.ys[:, 2], label='Lymph Node', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Blood Compartments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lung compartments (indices 3-8)
    ax2 = fig1.add_subplot(gs[0, 1])
    ax2.semilogy(sol.ts, sol.ys[:, 3], label='Lung Plasma', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 4], label='Lung Blood Cells', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 5], label='Lung ISF', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 6], label='Lung E Unbound', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 7], label='Lung E Bound', linewidth=2)
    ax2.semilogy(sol.ts, sol.ys[:, 8], label='Lung FcRn Free', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('Lung Compartments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Brain compartments (indices 9-13)
    ax3 = fig1.add_subplot(gs[0, 2])
    ax3.semilogy(sol.ts, sol.ys[:, 9], label='Brain Plasma', linewidth=2)
    ax3.semilogy(sol.ts, sol.ys[:, 10], label='BBB Unbound', linewidth=2)
    ax3.semilogy(sol.ts, sol.ys[:, 11], label='BBB Bound', linewidth=2)
    ax3.semilogy(sol.ts, sol.ys[:, 12], label='Brain ISF', linewidth=2)
    ax3.semilogy(sol.ts, sol.ys[:, 13], label='Brain Blood Cells', linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Concentration')
    ax3.set_title('Brain Compartments')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # CSF compartments (indices 14-19)
    ax4 = fig1.add_subplot(gs[1, 0])
    ax4.semilogy(sol.ts, sol.ys[:, 14], label='BCSFB Unbound', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 15], label='BCSFB Bound', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 16], label='Lateral Ventricle', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 17], label='Third/Fourth Ventricle', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 18], label='Cisterna Magna', linewidth=2)
    ax4.semilogy(sol.ts, sol.ys[:, 19], label='Subarachnoid Space', linewidth=2)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Concentration')
    ax4.set_title('CSF Compartments')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Liver compartments (indices 20-25)
    ax5 = fig1.add_subplot(gs[1, 1])
    ax5.semilogy(sol.ts, sol.ys[:, 20], label='Liver Plasma', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 21], label='Liver Blood Cells', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 22], label='Liver ISF', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 23], label='Liver E Unbound', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 24], label='Liver E Bound', linewidth=2)
    ax5.semilogy(sol.ts, sol.ys[:, 25], label='Liver FcRn Free', linewidth=2)
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('Concentration')
    ax5.set_title('Liver Compartments')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Optional: Add a text box with final concentrations in the empty spot
    ax6 = fig1.add_subplot(gs[1, 2])
    ax6.axis('off')
    final_conc_text = "Final Concentrations:\n\n"
    for i, comp in enumerate(['Blood Plasma', 'Blood Cells', 'Lymph Node']):
        final_conc_text += f"{comp}: {sol.ys[-1, i]:.6f}\n"
    ax6.text(0, 0.9, final_conc_text, fontsize=8, verticalalignment='top')

    # Create a second figure for all curves
    fig2 = plt.figure(figsize=(15, 8))
    ax = fig2.add_subplot(111)
    
    # Plot all curves with labels
    labels = [
        # Blood compartments
        'Blood Plasma', 'Blood Cells', 'Lymph Node',
        # Lung compartments
        'Lung Plasma', 'Lung Blood Cells', 'Lung ISF',
        'Lung E Unbound', 'Lung E Bound', 'Lung FcRn Free',
        # Brain compartments
        'Brain Plasma', 'BBB Unbound', 'BBB Bound',
        'Brain ISF', 'Brain Blood Cells',
        # CSF compartments
        'BCSFB Unbound', 'BCSFB Bound', 'Lateral Ventricle',
        'Third/Fourth Ventricle', 'Cisterna Magna', 'Subarachnoid Space',
        # Liver compartments
        'Liver Plasma', 'Liver Blood Cells', 'Liver ISF',
        'Liver E Unbound', 'Liver E Bound', 'Liver FcRn Free'
    ]
    
    for i, label in enumerate(labels):
        ax.semilogy(sol.ts, sol.ys[:, i], label=label, linewidth=1)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Concentration')
    ax.set_title('All Compartments')
    ax.grid(True, alpha=0.3)
    
    # Add legend outside of plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save both figures
    fig1.savefig('concentration_plots_compartments.png', dpi=300, bbox_inches='tight')
    fig2.savefig('concentration_plots_all.png', dpi=300, bbox_inches='tight')
    
    plt.show()

    # Print final concentrations
    print("\nFinal Concentrations:")
    print("-" * 30)
    compartments = [
        'Blood:', 
        '  Plasma', '  Blood Cells', '  Lymph Node',
        'Lung:', 
        '  Plasma', '  Blood Cells', '  ISF', 
        '  Endosomal Unbound', '  Endosomal Bound', '  FcRn Free',
        'Brain:', 
        '  Plasma', '  BBB Unbound', '  BBB Bound',
        '  ISF', '  Blood Cells',
        'CSF:',
        '  BCSFB Unbound', '  BCSFB Bound', '  Lateral Ventricle', 
        '  Third/Fourth Ventricle', '  Cisterna Magna', '  Subarachnoid Space',
        'Liver:',
        '  Plasma', '  Blood Cells', '  ISF',
        '  Endosomal Unbound', '  Endosomal Bound', '  FcRn Free'
    ]

    for i, comp in enumerate(compartments):
        if comp.endswith(':'):
            print(f"\n{comp}")
        else:
            print(f"{comp:<25}: {sol.ys[-1, i-1]:.6f}")

if __name__ == "__main__":
    sol, registry = run_simulation()
    plot_results(sol, registry) 