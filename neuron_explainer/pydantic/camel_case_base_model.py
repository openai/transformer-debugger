from pydantic import BaseModel


def to_camel(string: str) -> str:
    return "".join(word.capitalize() if i > 0 else word for i, word in enumerate(string.split("_")))


class CamelCaseBaseModel(BaseModel):
    """
    Base model that will automatically generate camelCase aliases for fields. Python code can use
    either snake_case or camelCase names. When Typescript code is generated, it will only use the
    camelCase names.
    """

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True
