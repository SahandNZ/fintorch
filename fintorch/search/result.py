from typing import Dict, Any


class SearchResult:
    def __init__(self, inputs: Dict[str, Any], objective: float):
        self.__inputs: Dict[str, Any] = inputs
        self.__objective: float = objective

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.__inputs

    @property
    def objective(self) -> float:
        return self.__objective

    def __str__(self):
        return f"{self.inputs} => {self.objective}"
