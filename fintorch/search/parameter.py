from typing import List, Any


class SearchParameter:
    def __init__(self, name: str, values: List[Any]):
        self.__name: str = name
        self.__values: List[Any] = values

    @property
    def name(self) -> str:
        return self.__name

    @property
    def values(self) -> List[Any]:
        return self.__values
