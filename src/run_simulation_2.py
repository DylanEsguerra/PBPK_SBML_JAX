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

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from src.main_2 import main, load_module_params

def setup_coupled_system(registry, models, all_params):
    """Set up the coupled system using registry configuration"""
    
    # Calculate state indices for each module
    state_indices = {}
    current_index = 0
    for module in registry.get_execution_order():
        n_states = len(models[module].y0)
        state_indices[module] = (current_index, current_index + n_states)
        current_index += n_states

    @jit
    def coupled_system(t, y, args):
        # Split states into modules
        states = {}
        for module in registry.get_execution_order():
            start, end = state_indices[module]
            states[module] = y[start:end]
        
        # Initialize parameters
        params = {module: models[module].c.copy() for module in registry.modules}
        
        # Update coupling parameters
        for module in registry.get_execution_order():
            config = registry.modules[module]
            for param, source in config.coupling_params.items():
                if source in states:
                    source_idx = models[source].y_indexes.get(param)
                    target_idx = models[module].c_indexes.get(param)
                    if source_idx is not None and target_idx is not None:
                        params[module] = params[module].at[target_idx].set(
                            states[source][source_idx]
                        )
        
        # Calculate derivatives in execution order
        derivatives = []
        for module in registry.get_execution_order():
            dy_dt = models[module].RateofSpeciesChange()(
                states[module], t, {}, params[module]
            )
            # Handle NaNs by replacing them with zeros
            dy_dt = jnp.nan_to_num(dy_dt, nan=0.0)
            derivatives.append(dy_dt)
            
        return jnp.concatenate(derivatives)

    return coupled_system, state_indices

def run_simulation():
    # Initialize registry and process modules
    registry = main()
    
    # Import models and load parameters
    models = {}
    all_params = {}
    for module_name in registry.get_execution_order():
        config = registry.modules[module_name]
        model_module = importlib.import_module(f"generated.jax.{module_name}_jax")
        importlib.reload(model_module)
        models[module_name] = model_module
        all_params[module_name] = load_module_params(config)

    # Set up coupled system
    coupled_system, state_indices = setup_coupled_system(registry, models, all_params)

    # Combine initial conditions
    y0_list = []
    for module_name in registry.get_execution_order():
        y0_list.append(models[module_name].y0)
    y0_combined = jnp.concatenate(y0_list)

    # Set up solver
    t0, t1 = 0.0, 2000.0
    dt0 = 0.01
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 10000))
    solver = diffrax.Heun()
    term = diffrax.ODETerm(coupled_system)

    # Solve system
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t0, t1=t1, dt0=dt0,
        y0=y0_combined,
        args=(None,),
        saveat=saveat,
        max_steps=None,
        stepsize_controller=diffrax.ConstantStepSize()
    )

    return sol, registry, state_indices, models

def plot_results(sol, registry, state_indices, models):
    # Create figure with dynamic grid layout
    n_modules = len(registry.modules)
    n_rows = (n_modules + 2) // 3  # Ceiling division
    fig = plt.figure(figsize=(15, 5 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.3, wspace=0.3)

    # Plot each module's results
    for i, module_name in enumerate(registry.get_execution_order()):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Get state indices for this module
        start_idx, end_idx = state_indices[module_name]
        
        # Plot each state variable
        for j in range(start_idx, end_idx):
            state_name = list(models[module_name].y_indexes.keys())[j - start_idx]
            ax.plot(sol.ts, sol.ys[:, j], label=state_name, linewidth=2)
            
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{module_name.capitalize()} Compartments')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.savefig('concentration_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print final concentrations
    print("\nFinal Concentrations:")
    print("-" * 30)
    
    for module_name in registry.get_execution_order():
        start_idx, end_idx = state_indices[module_name]
        print(f"\n{module_name.capitalize()}:")
        for j in range(start_idx, end_idx):
            state_name = list(models[module_name].y_indexes.keys())[j - start_idx]
            print(f"  {state_name:<25}: {sol.ys[-1, j]:.6f}")

if __name__ == "__main__":
    sol, registry, state_indices, models = run_simulation()
    plot_results(sol, registry, state_indices, models) 