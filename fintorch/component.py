from abc import ABC


class Component(ABC):
    def __init__(self, name: str, short_name: str, description: str):
        self.__name: str = name
        self.__short_name: str = short_name
        self.__description: str = description

    @property
    def name(self) -> str:
        return self.__name

    @property
    def short_name(self) -> str:
        return self.__short_name

    @property
    def description(self) -> str:
        return self.__description
