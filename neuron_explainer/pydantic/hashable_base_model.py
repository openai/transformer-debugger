from .camel_case_base_model import CamelCaseBaseModel


class HashableBaseModel(CamelCaseBaseModel):
    def __hash__(self) -> int:
        values = tuple(getattr(self, name) for name in self.__annotations__.keys())
        # Convert lists to tuples.
        values = tuple(value if not isinstance(value, list) else tuple(value) for value in values)
        return hash(values)
