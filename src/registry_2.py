from dataclasses import dataclass
from typing import List, Dict, Set
from pathlib import Path

@dataclass
class ModuleConfig:
    """Configuration for each module (blood, brain, csf, etc.)"""
    name: str
    param_sheets: List[str]
    dependencies: List[str]
    xml_path: Path
    jax_path: Path
    initial_conditions: List[str]
    coupling_params: Dict[str, str]  # Maps parameter name to source module

class ModuleRegistry:
    def __init__(self):
        self.modules: Dict[str, ModuleConfig] = {}
    
    def register_module(self, config: ModuleConfig):
        """Register a new module with the system"""
        self.modules[config.name] = config
    
    def find_cycles(self) -> List[List[str]]:
        """Find all dependency cycles in the module graph"""
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node: str):
            if node in path_set:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            path_set.add(node)
            
            for dep in self.modules[node].dependencies:
                if dep in self.modules:
                    dfs(dep)
            
            path.pop()
            path_set.remove(node)
        
        for name in self.modules:
            dfs(name)
        
        return cycles

    def get_strongly_connected_components(self) -> List[List[str]]:
        """Find strongly connected components (groups of circular dependencies)"""
        index_counter = [0]
        index = {}
        lowlink = {}
        stack = []
        on_stack = set()
        components = []

        def strongconnect(node: str):
            index[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            for dep in self.modules[node].dependencies:
                if dep not in index:
                    strongconnect(dep)
                    lowlink[node] = min(lowlink[node], lowlink[dep])
                elif dep in on_stack:
                    lowlink[node] = min(lowlink[node], index[dep])

            if lowlink[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                if len(component) > 1:  # Only include non-trivial components
                    components.append(component)

        for node in self.modules:
            if node not in index:
                strongconnect(node)

        return components

    def get_execution_order(self) -> List[str]:
        """Returns modules grouped by their dependency relationships"""
        # Find strongly connected components (circular dependencies)
        components = self.get_strongly_connected_components()
        
        # Create a set of all modules in circular dependencies
        circular_modules = {module for component in components for module in component}
        
        # Get remaining modules (those not in circular dependencies)
        remaining_modules = set(self.modules.keys()) - circular_modules
        
        # Create execution order with circular dependencies grouped together
        order = []
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            if name not in self.modules:
                raise ValueError(f"Unknown module: {name}")
            
            # If module is part of a circular dependency group
            if name in circular_modules:
                # Add all modules in the same component if not already visited
                for component in components:
                    if name in component:
                        for module in component:
                            if module not in visited:
                                visited.add(module)
                                order.append(module)
                        break
            else:
                # Process normal dependencies first
                for dep in self.modules[name].dependencies:
                    visit(dep)
                visited.add(name)
                order.append(name)
        
        # Process all modules
        for name in self.modules:
            visit(name)
            
        return order