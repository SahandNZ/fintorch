from abc import ABC, abstractmethod
from typing import Callable, List, Any, Dict

from fintorch.search.parameter import SearchParameter
from fintorch.search.result import SearchResult


class Search(ABC):
    def __init__(self, parameters: List[SearchParameter], objective_function: Callable, maximize: bool):
        self.__parameters: List[SearchParameter] = parameters
        self.__objective_function: Callable = objective_function
        self.__maximize: bool = maximize

    @property
    def parameters(self) -> List[SearchParameter]:
        return self.__parameters

    @property
    def objective_function(self) -> Callable:
        return self.__objective_function

    @property
    def maximize(self) -> bool:
        return self.__maximize

    def search(self) -> List[SearchResult]:
        results = self._search()
        results = sorted(results, key=lambda result: result.objective, reverse=self.maximize)
        return results

    @abstractmethod
    def _input_generator(self) -> List[Dict[str, Any]]:
        raise NotImplemented()

    @abstractmethod
    def _search(self) -> List[SearchResult]:
        raise NotImplemented()
