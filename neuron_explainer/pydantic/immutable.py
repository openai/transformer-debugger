from typing import TypeVar

from pydantic import BaseConfig, BaseModel

T = TypeVar("T", bound=BaseModel)


def immutable(cls: type[T]) -> type[T]:
    """
    Makes a Pydantic model immutable.

    Annotate a Pydantic class with `@immutable` to prevent the values of its fields from being
    changed after an instance is constructed. (This only guarantees shallow immutability of course:
    fields may have their internal state change.)
    """

    class Config(BaseConfig):
        frozen: bool = True

    cls.Config = Config
    return cls
