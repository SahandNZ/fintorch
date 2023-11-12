import itertools
from typing import List, Callable, Any, Dict

from fintorch.search.parameter import SearchParameter
from fintorch.search.result import SearchResult
from fintorch.search.search import Search


class GridSearch(Search):
    def __init__(self, parameters: List[SearchParameter], objective_function: Callable, maximize: bool = True):
        super().__init__(parameters, objective_function, maximize)

    def _input_generator(self) -> List[Dict[str, Any]]:
        names = [param.name for param in self.parameters]
        search_space = [param.values for param in self.parameters]
        for values in itertools.product(*search_space):
            inputs = {name: value for name, value in zip(names, values)}
            yield inputs

    def _search(self) -> List[SearchResult]:
        results = []
        for inputs in self._input_generator():
            objective = self.objective_function(**inputs)
            result = SearchResult(inputs=inputs, objective=objective)
            results.append(result)

        return results
