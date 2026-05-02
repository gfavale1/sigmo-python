import sigmo
from rdkit import Chem


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


def sigmo_has_match(query: str, target: str) -> bool:
    result = sigmo.match(
        query=query,
        target=target,
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
    )

    return result.total_matches > 0


def test_sigmo_matches_rdkit_on_basic_cases():
    for query, target, expected in CASES:
        rdkit_result = rdkit_has_match(query, target)
        sigmo_result = sigmo_has_match(query, target)

        assert rdkit_result == expected
        assert sigmo_result == expected
        assert sigmo_result == rdkit_result