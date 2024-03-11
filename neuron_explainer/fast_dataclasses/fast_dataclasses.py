# Utilities for dataclasses that are very fast to serialize and deserialize, with limited data
# validation. Fields must not be tuples, since they get serialized and then deserialized as lists.
#
# The unit tests for this library show how to use it.

import json
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial
from typing import Any

import orjson

dataclasses_by_name = {}
# TODO(williamrs): remove deserialization using fieldnames once all data is in the new format
dataclasses_by_fieldnames = {}


@dataclass
class FastDataclass:
    dataclass_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.dataclass_name = self.__class__.__name__

    @classmethod
    def field_renamed(cls, old_name: str, new_name: str) -> None:
        if not hasattr(cls, "field_renames"):
            cls.field_renames = {}  # type: ignore
        assert old_name not in cls.field_renames, f"{old_name} already renamed"  # type: ignore
        existing_field_names = [f.name for f in fields(cls)]
        assert new_name in existing_field_names, f"{new_name} not in {existing_field_names}"
        assert old_name not in existing_field_names, f"{old_name} still in {existing_field_names}"
        cls.field_renames[old_name] = new_name  # type: ignore

    # Type checking is suppressed in these two functions because mypy doesn't like that we're
    # setting attributes at runtime. We need to do that because if we defined the attributes at the
    # class level, they'd be shared across subclasses.
    @classmethod
    def field_deleted(cls, field_name: str) -> None:
        if not hasattr(cls, "deleted_fields"):
            cls.deleted_fields = []  # type: ignore
        assert field_name not in cls.deleted_fields, f"{field_name} already deleted"  # type: ignore
        existing_field_names = [f.name for f in fields(cls)]
        assert field_name not in existing_field_names, f"{field_name} still in use"
        cls.deleted_fields.append(field_name)  # type: ignore

    @classmethod
    def was_previously_named(cls, old_name: str) -> None:
        assert old_name not in dataclasses_by_name, f"{old_name} still in use as a dataclass name"
        dataclasses_by_name[old_name] = cls
        name_set = frozenset(f.name for f in fields(cls) if f.name != "dataclass_name")
        dataclasses_by_fieldnames[name_set] = cls


def register_dataclass(cls):  # type: ignore
    assert is_dataclass(cls), "Only dataclasses can be registered."
    dataclasses_by_name[cls.__name__] = cls
    name_set = frozenset(f.name for f in fields(cls) if f.name != "dataclass_name")
    dataclasses_by_fieldnames[name_set] = cls
    return cls


def dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)


def _object_hook(d: Any, backwards_compatible: bool = True) -> Any:
    # If d is a list, recurse.
    if isinstance(d, list):
        return [_object_hook(x, backwards_compatible=backwards_compatible) for x in d]
    # If d is not a dict, return it as is.
    if not isinstance(d, dict):
        return d
    cls = None
    if "dataclass_name" in d:
        if d["dataclass_name"] in dataclasses_by_name:
            cls = dataclasses_by_name[d["dataclass_name"]]
        else:
            assert backwards_compatible, (
                f"Dataclass {d['dataclass_name']} not found, set backwards_compatible=True if you "
                f"are okay with that."
            )
    # Load objects created without dataclass_name set.
    else:
        # Try our best to find a dataclass if backwards_compatible is True.
        if backwards_compatible:
            d_fields = frozenset(d.keys())
            if d_fields in dataclasses_by_fieldnames:
                cls = dataclasses_by_fieldnames[d_fields]
            elif len(d_fields) > 0:
                # Check if the fields are a subset of a dataclass (if the dataclass had extra fields
                # added since the data was created). Note that this will fail if fields were removed
                # from the dataclass.
                for key, possible_cls in dataclasses_by_fieldnames.items():
                    if d_fields.issubset(key):
                        cls = possible_cls
                        break
                else:
                    print(f"Could not find dataclass for {d_fields} {cls}")

    if cls is not None:
        # Apply renames and deletions.
        if hasattr(cls, "field_renames"):
            d = {cls.field_renames.get(k, k): v for k, v in d.items()}
        if hasattr(cls, "deleted_fields"):
            d = {k: v for k, v in d.items() if k not in cls.deleted_fields}

    new_d = {
        k: _object_hook(v, backwards_compatible=backwards_compatible)
        for k, v in d.items()
        if k != "dataclass_name"
    }
    if cls is not None:
        return cls(**new_d)
    else:
        return new_d


def loads(s: str | bytes, backwards_compatible: bool = True) -> Any:
    return json.loads(
        s,
        object_hook=partial(_object_hook, backwards_compatible=backwards_compatible),
    )
