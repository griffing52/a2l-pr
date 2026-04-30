from abc import ABC, abstractmethod
from typing import Dict, Any

class DataAdapter(ABC):
    """Base class for trajectory data adapters."""
    
    @abstractmethod
    def load(self, source: Any) -> Dict:
        """Load trajectory data into the standard dictionary format.
        
        Returns:
            Dict containing 'observations', 'actions', and optional 'metadata'.
        """
        pass
        
    @abstractmethod
    def save(self, trajectory: Dict, destination: Any) -> None:
        """Save standard trajectory dictionary to the destination format."""
        pass
