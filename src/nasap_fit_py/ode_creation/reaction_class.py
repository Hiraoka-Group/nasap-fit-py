from collections.abc import Iterable, Sequence
from typing import Generic, TypeVar

_T_co = TypeVar('_T_co', covariant=True)  # type of assembly
_S_co = TypeVar('_S_co', covariant=True)  # type of reaction kind


class Reaction(Generic[_T_co, _S_co]):
    def __init__(
            self, reactants: Iterable[_T_co], products: Iterable[_T_co],
            reaction_kind: _S_co, duplicate_count: int
            ) -> None:
        # If two A assemblies are consumed, then two As should be in the list.
        self._reactants = tuple(reactants)
        # If two B assemblies are produced, then two Bs should be in the list.
        self._products = tuple(products)
        self._duplicate_count = duplicate_count
        self._reaction_kind = reaction_kind

    @property
    def reactants(self) -> tuple[_T_co, ...]:
        return self._reactants

    @property
    def products(self) -> tuple[_T_co, ...]:
        return self._products

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def reaction_kind(self) -> _S_co:
        return self._reaction_kind

    def __hash__(self) -> int:
        return hash((
            self._reactants, self._products, self._duplicate_count,
            self._reaction_kind))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reaction):
            return False
        return (
            self._reactants == other._reactants and
            self._products == other._products and
            self._duplicate_count == other._duplicate_count and
            self._reaction_kind == other._reaction_kind
            )
