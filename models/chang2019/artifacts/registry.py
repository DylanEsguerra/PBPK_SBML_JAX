from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

@dataclass
class ModuleConfig:
    """Configuration for each module (blood, brain, csf, etc.)"""
    name: str
    param_sheets: List[str]
    dependencies: List[str]
    xml_path: Path
    jax_path: Path
    solver_path: Path
    initial_conditions: List[str]
    coupling_params: Dict[str, str]  # Maps parameter name to source module

class ModuleRegistry:
    def __init__(self):
        self.modules: Dict[str, ModuleConfig] = {}
    
    def register_module(self, config: ModuleConfig):
        """Register a new module with the system"""
        self.modules[config.name] = config
    
    def get_execution_order(self) -> List[str]:
        """Returns modules in correct dependency order"""
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited:
                return
            if name not in self.modules:
                raise ValueError(f"Unknown module: {name}")
            
            # First visit all dependencies
            for dep in self.modules[name].dependencies:
                visit(dep)
                
            visited.add(name)
            order.append(name)
            
        for name in self.modules:
            visit(name)
            
        return order 