from typing import Dict, Type, List, Optional
from .interface import BaseProcessor

class ProcessorRegistry:
    _processors: Dict[str, Type[BaseProcessor]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a processor class with a given name."""
        def decorator(processor_cls: Type[BaseProcessor]):
            cls._processors[name] = processor_cls
            return processor_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseProcessor]]:
        """Get a processor class by name."""
        return cls._processors.get(name)
        
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered processor names."""
        return list(cls._processors.keys())

