import json
from dataclasses import dataclass

import pytest

from .fast_dataclasses import FastDataclass, dumps, loads, register_dataclass


# Inheritance is a bit tricky with our setup. dataclass_name must be set for instances of these
# classes to serialize and deserialize correctly, but if it's given a default value, then subclasses
# can't have any fields that don't have default values, because of how constructors are generated
# for dataclasses (fields with no default value can't follow those with default values). To work
# around this, we set dataclass_name in __post_init__ on the base class, which is called after the
# constructor. The implementation does the right thing for both the base class and the subclass.
@register_dataclass
@dataclass
class DataclassC(FastDataclass):
    ints: list[int]


@register_dataclass
@dataclass
class DataclassC_ext(DataclassC):
    s: str


@register_dataclass
@dataclass
class DataclassB(FastDataclass):
    str_to_c: dict[str, DataclassC]
    cs: list[DataclassC]


@register_dataclass
@dataclass
class DataclassA(FastDataclass):
    floats: list[float]
    strings: list[str]
    bs: list[DataclassB]


@register_dataclass
@dataclass
class DataclassD(FastDataclass):
    s1: str
    s2: str = "default"


def test_dataclasses() -> None:
    a = DataclassA(
        floats=[1.0, 2.0],
        strings=["a", "b"],
        bs=[
            DataclassB(
                str_to_c={"a": DataclassC(ints=[1, 2]), "b": DataclassC(ints=[3, 4])},
                cs=[DataclassC(ints=[5, 6]), DataclassC_ext(ints=[7, 8], s="s")],
            ),
            DataclassB(
                str_to_c={"c": DataclassC_ext(ints=[9, 10], s="t"), "d": DataclassC(ints=[11, 12])},
                cs=[DataclassC(ints=[13, 14]), DataclassC(ints=[15, 16])],
            ),
        ],
    )
    assert loads(dumps(a)) == a


def test_c_and_c_ext() -> None:
    c_ext = DataclassC_ext(ints=[3, 4], s="s")
    assert loads(dumps(c_ext)) == c_ext

    c = DataclassC(ints=[1, 2])
    assert loads(dumps(c)) == c


def test_bad_serialized_data() -> None:
    assert type(loads(dumps(DataclassC(ints=[3, 4])))) == DataclassC
    assert type(loads('{"ints": [3, 4]}', backwards_compatible=False)) == dict
    assert type(loads('{"ints": [3, 4], "dataclass_name": "DataclassC"}')) == DataclassC
    with pytest.raises(TypeError):
        loads('{"ints": [3, 4], "bogus_extra_field": "foo", "dataclass_name": "DataclassC"}')
    with pytest.raises(TypeError):
        loads('{"ints_field_is_missing": [3, 4], "dataclass_name": "DataclassC"}')
    assert type(loads('{"s1": "test"}', backwards_compatible=False)) == dict
    assert type(loads('{"s1": "test"}', backwards_compatible=True)) == DataclassD


def test_renames_and_deletes() -> None:
    @register_dataclass
    @dataclass
    class FooV1(FastDataclass):
        a: int
        b: str
        c: float

    # Version with c renamed.
    @register_dataclass
    @dataclass
    class FooV2(FastDataclass):
        a: int
        b: str
        c_renamed: float

    # Version with c renamed and b deleted.
    @register_dataclass
    @dataclass
    class FooV3(FastDataclass):
        a: int
        c_renamed: float

    # Basic sanity checks.
    foo_v1 = FooV1(a=1, b="hello", c=3.14)
    foo_v1_bytes = dumps(foo_v1)
    assert loads(foo_v1_bytes) == foo_v1

    # Deserializing a FooV1 as a FooV2 should fail.
    foo_v1_dict = json.loads(foo_v1_bytes)
    # Change the dataclass_name so this will be interpreted as a FooV2.
    foo_v1_dict["dataclass_name"] = "FooV2"
    foo_v1_as_v2_bytes = dumps(foo_v1_dict)
    with pytest.raises(TypeError):
        loads(foo_v1_as_v2_bytes)

    # Register the field's rename, then try again. This time it should succeed.
    FooV2.field_renamed("c", "c_renamed")
    foo_v1_as_v2 = loads(foo_v1_as_v2_bytes)
    assert type(foo_v1_as_v2) == FooV2
    assert foo_v1_as_v2.a == foo_v1.a
    assert foo_v1_as_v2.b == foo_v1.b
    assert foo_v1_as_v2.c_renamed == foo_v1.c

    # Deserializing a FooV1 as a FooV3 should fail, even with the rename.
    FooV3.field_renamed("c", "c_renamed")
    # Change the dataclass_name so this will be interpreted as a FooV3.
    foo_v1_dict["dataclass_name"] = "FooV3"
    foo_v1_as_v3_bytes = dumps(foo_v1_dict)
    with pytest.raises(TypeError):
        loads(foo_v1_as_v3_bytes)

    # Register the field's deletion, then try again. This time it should succeed.
    FooV3.field_deleted("b")
    foo_v1_as_v3 = loads(foo_v1_as_v3_bytes)
    assert type(foo_v1_as_v3) == FooV3
    assert foo_v1_as_v3.a == foo_v1.a
    assert foo_v1_as_v3.c_renamed == foo_v1.c


def test_class_rename() -> None:
    @register_dataclass
    @dataclass
    class BarRenamed(FastDataclass):
        d: str
        e: int

    BarRenamed.was_previously_named("Bar")

    # We should be able to deserialize an old Bar as a BarRenamed.
    bar = BarRenamed(d="hello", e=1)
    bar_bytes = dumps(bar)
    bar_dict = json.loads(bar_bytes)
    # Change the dataclass_name so this looks like an old Bar.
    bar_dict["dataclass_name"] = "Bar"
    bar_as_bar_bytes = dumps(bar_dict)
    bar_as_bar_renamed = loads(bar_as_bar_bytes)
    assert type(bar_as_bar_renamed) == BarRenamed
    assert bar_as_bar_renamed.d == bar.d
    assert bar_as_bar_renamed.e == bar.e


def test_invalid_renames_and_deletions() -> None:
    @register_dataclass
    @dataclass
    class OldBaz(FastDataclass):
        a: int
        b: str
        c: float

    @register_dataclass
    @dataclass
    class Baz(FastDataclass):
        a: int
        b: str
        c: float

    # Renaming a field should fail if the new name doesn't exist.
    with pytest.raises(AssertionError):
        Baz.field_renamed("d", "d_renamed")

    # Renaming a field should fail if the old name is still in use.
    with pytest.raises(AssertionError):
        Baz.field_renamed("a", "b")

    # Deleting a field should fail if the name is still in use.
    with pytest.raises(AssertionError):
        Baz.field_deleted("a")

    # Renaming a dataclass should fail if the old name is still in use.
    with pytest.raises(AssertionError):
        Baz.was_previously_named("OldBaz")

    # Renaming the same field twice should fail.
    Baz.field_renamed("old_c", "c")
    with pytest.raises(AssertionError):
        Baz.field_renamed("old_c", "b")

    # Deleting a field twice should fail.
    Baz.field_deleted("z")
    with pytest.raises(AssertionError):
        Baz.field_deleted("z")
