import pytest

from src.nasap_fit_py.ode_creation import Reaction, load_reactions_from_csv


@pytest.fixture
def default_header() -> str:
    return (
        'init_assem_id,entering_assem_id,'
        'product_assem_id,leaving_assem_id,'
        'duplicate_count,reaction_kind\n'
    )


# Helper function to test attributes of Reaction
def assert_reaction_properties(
        reaction: Reaction, 
        reactants, products, reaction_kind, duplicate_count):
    assert reaction.reactants == reactants
    assert reaction.products == products
    assert reaction.reaction_kind == reaction_kind
    assert reaction.duplicate_count == duplicate_count


def test(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    csv.write_text(
        default_header +
        '0,1,2,3,4,0\n'
        '1,2,3,4,5,1\n'
    )
    reactions = load_reactions_from_csv(csv)

    assert len(reactions) == 2
    assert_reaction_properties(reactions[0], (0, 1), (2, 3), 0, 4)
    assert_reaction_properties(reactions[1], (1, 2), (3, 4), 1, 5)

# ==================================================
# Test data with empty values

# Data with no init_assem_id is allowed 
# as long as entering_assem_id is provided.
def test_no_init_assem(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # init_assem_id is empty
    csv.write_text(default_header + ',1,2,3,4,0\n')
    reactions = load_reactions_from_csv(csv)

    assert len(reactions) == 1
    # init_assem_id is empty, so reactants should be [1]
    assert_reaction_properties(reactions[0], (1,), (2, 3), 0, 4)


def test_no_entering_assem(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # entering_assem_id is empty
    csv.write_text(default_header + '0,,2,3,4,0\n')
    reactions = load_reactions_from_csv(csv)
    
    assert len(reactions) == 1
    # entering_assem_id is empty, so reactants should be [0]
    assert_reaction_properties(reactions[0], (0,), (2, 3), 0, 4)


def test_no_init_or_entering_assem(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # Both init_assem_id and entering_assem_id are empty
    csv.write_text(default_header + ',,2,3,4,0\n')
    # At least one reactant must be provided.
    with pytest.raises(ValueError):
        load_reactions_from_csv(csv)


def test_no_product_assem(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # product_assem_id is empty
    csv.write_text(default_header + '0,1,,3,4,0\n')
    reactions = load_reactions_from_csv(csv)
    
    assert len(reactions) == 1
    # product_assem_id is empty, so products should be [3]
    assert_reaction_properties(reactions[0], (0, 1), (3,), 0, 4)


def test_no_leaving_assem(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # leaving_assem_id is empty
    csv.write_text(default_header + '0,1,2,,4,0\n')
    reactions = load_reactions_from_csv(csv)
    
    assert len(reactions) == 1
    # leaving_assem_id is empty, so products should be [2]
    assert_reaction_properties(reactions[0], (0, 1), (2,), 0, 4)


def test_no_product_or_leaving_assem(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # Both product_assem_id and leaving_assem_id are empty
    csv.write_text(default_header + '0,1,,,4,0\n')
    # At least one product must be provided.
    with pytest.raises(ValueError):
        load_reactions_from_csv(csv)


def test_no_duplicate_count(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # duplicate_count is empty
    csv.write_text(default_header + '0,1,2,3,,0\n')
    # Duplicate count must be provided.
    with pytest.raises(ValueError):
        load_reactions_from_csv(csv)


def test_no_reaction_kind(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    # reaction_kind is empty
    csv.write_text(default_header + '0,1,2,3,4,\n')
    # Reaction kind must be provided.
    with pytest.raises(ValueError):
        load_reactions_from_csv(csv)


# ==================================================
# Test custom column names

def test_custom_column_names(tmp_path):
    csv = tmp_path / 'reactions.csv'
    csv.write_text(
        'init,enter,product,leave,dup,kind\n'
        '0,1,2,3,4,0\n'
    )
    reactions = load_reactions_from_csv(
        csv,
        reactant_cols=('init', 'enter'),
        product_cols=('product', 'leave'),
        duplicate_count_col='dup',
        reaction_kind_col='kind',
    )
    
    assert len(reactions) == 1
    assert_reaction_properties(reactions[0], (0, 1), (2, 3), 0, 4)


# ==================================================
# Test data types

def test_int_assem_ids(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    csv.write_text(
        default_header +
        '0,1,2,3,4,0\n'
        '1,2,3,4,5,1\n'
    )
    reactions = load_reactions_from_csv(csv, assem_dtype='int')

    assert len(reactions) == 2
    assert_reaction_properties(reactions[0], (0, 1), (2, 3), 0, 4)
    assert_reaction_properties(reactions[1], (1, 2), (3, 4), 1, 5)


def test_str_assem_ids(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    csv.write_text(
        default_header +
        '0,1,2,3,4,0\n'
        '1,2,3,4,5,1\n'
    )
    reactions = load_reactions_from_csv(
        csv, assem_dtype='str', reaction_kind_dtype='str')

    assert len(reactions) == 2
    assert_reaction_properties(reactions[0], ('0', '1'), ('2', '3'), '0', 4)
    assert_reaction_properties(reactions[1], ('1', '2'), ('3', '4'), '1', 5)


def test_int_reaction_kind(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    csv.write_text(
        default_header +
        '0,1,2,3,4,0\n'
        '1,2,3,4,5,1\n'
    )
    reactions = load_reactions_from_csv(csv, reaction_kind_dtype='int')

    assert len(reactions) == 2
    assert_reaction_properties(reactions[0], (0, 1), (2, 3), 0, 4)
    assert_reaction_properties(reactions[1], (1, 2), (3, 4), 1, 5)


def test_str_reaction_kind(tmp_path, default_header):
    csv = tmp_path / 'reactions.csv'
    csv.write_text(
        default_header +
        '0,1,2,3,4,0\n'
        '1,2,3,4,5,1\n'
    )
    reactions = load_reactions_from_csv(csv, reaction_kind_dtype='str')

    assert len(reactions) == 2
    assert_reaction_properties(reactions[0], (0, 1), (2, 3), '0', 4)
    assert_reaction_properties(reactions[1], (1, 2), (3, 4), '1', 5)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
