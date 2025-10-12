from collections.abc import Sequence
from pathlib import Path
from typing import Literal, overload

import pandas as pd

from .reaction_class import Reaction


@overload
def load_reactions_from_csv(
        path: Path | str,
        *,
        reactant_cols: tuple[str, str] = (
            'init_assem_id', 'entering_assem_id'),
        product_cols: tuple[str, str] = (
            'product_assem_id', 'leaving_assem_id'),
        duplicate_count_col: str = 'duplicate_count',
        reaction_kind_col: str = 'reaction_kind',
        assem_dtype: Literal['int'] = 'int',
        reaction_kind_dtype: Literal['int'] = 'int'
        ) -> Sequence[Reaction[int, int]]:
    ...
@overload
def load_reactions_from_csv(
        path: Path | str,
        *,
        reactant_cols: tuple[str, str] = (
            'init_assem_id', 'entering_assem_id'),
        product_cols: tuple[str, str] = (
            'product_assem_id', 'leaving_assem_id'),
        duplicate_count_col: str = 'duplicate_count',
        reaction_kind_col: str = 'reaction_kind',
        assem_dtype: Literal['int'] = 'int',
        reaction_kind_dtype: Literal['str'] = 'str'
        ) -> Sequence[Reaction[int, str]]:
    ...
@overload
def load_reactions_from_csv(
        path: Path | str,
        *,
        reactant_cols: tuple[str, str] = (
            'init_assem_id', 'entering_assem_id'),
        product_cols: tuple[str, str] = (
            'product_assem_id', 'leaving_assem_id'),
        duplicate_count_col: str = 'duplicate_count',
        reaction_kind_col: str = 'reaction_kind',
        assem_dtype: Literal['str'] = 'str',
        reaction_kind_dtype: Literal['int'] = 'int'
        ) -> Sequence[Reaction[str, int]]:
    ...
@overload
def load_reactions_from_csv(
        path: Path | str,
        *,
        reactant_cols: tuple[str, str] = (
            'init_assem_id', 'entering_assem_id'),
        product_cols: tuple[str, str] = (
            'product_assem_id', 'leaving_assem_id'),
        duplicate_count_col: str = 'duplicate_count',
        reaction_kind_col: str = 'reaction_kind',
        assem_dtype: Literal['str'] = 'str',
        reaction_kind_dtype: Literal['str'] = 'str'
        ) -> Sequence[Reaction[str, str]]:
    ...
def load_reactions_from_csv(
        path: Path | str,
        *,
        reactant_cols: tuple[str, str] = (
            'init_assem_id', 'entering_assem_id'),
        product_cols: tuple[str, str] = (
            'product_assem_id', 'leaving_assem_id'),
        duplicate_count_col: str = 'duplicate_count',
        reaction_kind_col: str = 'reaction_kind',
        assem_dtype: Literal['int', 'str'] = 'int',
        reaction_kind_dtype: Literal['int', 'str'] = 'int'
        ) -> Sequence[Reaction]:
    assem_dtype_map = {'int': pd.Int64Dtype(), 'str': pd.StringDtype()}
    reaction_kind_dtype_map = {'int': pd.Int64Dtype(), 'str': pd.StringDtype()}
    assem_dtype_obj = assem_dtype_map[assem_dtype]
    reaction_kind_dtype_obj = reaction_kind_dtype_map[reaction_kind_dtype]
    df = pd.read_csv(
        path,
        usecols=[*reactant_cols, *product_cols, 
                 duplicate_count_col, reaction_kind_col],
        dtype={reactant_cols[0]: assem_dtype_obj, 
               reactant_cols[1]: assem_dtype_obj,
               product_cols[0]: assem_dtype_obj, 
               product_cols[1]: assem_dtype_obj,
               duplicate_count_col: pd.Int64Dtype(), 
               reaction_kind_col: reaction_kind_dtype_obj
               }
        )
    
    assem_type = int if assem_dtype == 'int' else str
    reaction_kind_type = int if reaction_kind_dtype == 'int' else str
    
    reactions: list[Reaction] = []

    for row in df.itertuples(index=False):
        reactants = []
        if (reactant1 := getattr(row, reactant_cols[0])) is not pd.NA:
            reactants.append(assem_type(reactant1))
        if (reactant2 := getattr(row, reactant_cols[1])) is not pd.NA:
            reactants.append(assem_type(reactant2))
        if not reactants:
            raise ValueError('At least one reactant must be provided.')
        products = []
        if (product1 := getattr(row, product_cols[0])) is not pd.NA:
            products.append(assem_type(product1))
        if (product2 := getattr(row, product_cols[1])) is not pd.NA:
            products.append(assem_type(product2))
        if not products:
            raise ValueError('At least one product must be provided.')
        if (dup_count := getattr(row, duplicate_count_col)) is pd.NA:
            raise ValueError('Duplicate count must be provided.')
        else:
            dup_count = int(dup_count)
        if (reaction_kind := getattr(row, reaction_kind_col)) is pd.NA:
            raise ValueError('Reaction kind must be provided.')
        else:
            reaction_kind = reaction_kind_type(reaction_kind)
        reactions.append(
            Reaction(
                reactants=reactants,
                products=products,
                reaction_kind=reaction_kind,
                duplicate_count=dup_count,
            )
        )

    return reactions
