import pytest
import sigmo
from rdkit import Chem
from conftest import assert_match_result


CASES = [
    ("CC", "CCC", True),
    ("CO", "CCC", False),
    ("C=O", "CC(=O)O", True),
    ("N", "CCO", False),
    ("c1ccccc1", "CCOC(=O)c1ccccc1", True),
]


def rdkit_has_match(query: str, target: str) -> bool:
    query_mol = Chem.MolFromSmarts(query)
    if query_mol is None:
        query_mol = Chem.MolFromSmiles(query)

    target_mol = Chem.MolFromSmiles(target)
    if target_mol is None:
        target_mol = Chem.MolFromSmarts(target)

    assert query_mol is not None
    assert target_mol is not None
    return bool(target_mol.HasSubstructMatch(query_mol))


@pytest.mark.parametrize("query,target,expected", CASES)
def test_sigmo_matches_rdkit_on_basic_cases(query, target, expected):
    result = sigmo.match(
        query=query,
        target=target,
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
    )

    assert_match_result(result)
    assert (result.total_matches > 0) == expected
    assert rdkit_has_match(query, target) == expected


@pytest.mark.parametrize("query,target,expected", CASES)
def test_integrated_rdkit_validation(query, target, expected):
    result = sigmo.match(
        query=query,
        target=target,
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
        validate_with_rdkit=True,
    )

    assert_match_result(result)
    assert result.validation["enabled"] is True
    assert result.validation["method"] == "RDKit HasSubstructMatch"
    assert result.validation["checked_pairs"] == 1
    assert result.validation["agreements"] == 1
    assert result.validation["passed"] is True
    assert len(result.validation["disagreements"]) == 0
    assert (result.total_matches > 0) == expected