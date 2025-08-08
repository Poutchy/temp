"""Module for saving functions"""

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

T = TypeVar("T")


class BaseAdapter(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def save(obj: T) -> list[str]:
        pass

    @staticmethod
    @abstractmethod
    def load(lines: list[str]) -> T:
        pass


def save[U](obj: U, adapter: Type[BaseAdapter[U]], path: str):
    """
    Function for saving structures of the project

    :param obj: The saved object
    :type obj: U
    :param adapter: The saving adapter of the object
    :type adapter: Type[BaseAdapter[U]]
    :param path: The path of the saved object
    :type path: str
    """
    lines = adapter.save(obj)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load[U](adapter: Type[BaseAdapter[U]], path: str) -> U:
    """
    Function for loading structures of the project

    :param adapter: The saving adapter of the object
    :type adapter: Type[BaseAdapter[U]]
    :param path: The path of the loaded object
    :type path: str
    :returns: The loaded object
    :rtype: U
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return adapter.load(lines)
