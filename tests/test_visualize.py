import pytest

import sigmo
from sigmo.visualize import (
    draw_match_pair,
    draw_molecule,
    mol_from_input,
    to_networkx,
)


def test_mol_from_input_smiles():
    """
    SMILES strings should be converted into valid RDKit molecules.
    """
    mol = mol_from_input("CCO", input_format="smiles")

    assert mol is not None
    assert mol.GetNumAtoms() == 3


def test_mol_from_input_smarts():
    """
    SMARTS strings should be converted into valid RDKit query molecules.
    """
    mol = mol_from_input(
        "[CX3]=[OX1]",
        input_format="smarts",
        role="query",
    )

    assert mol is not None


def test_draw_molecule_returns_image():
    """
    draw_molecule() should return an image object for a valid molecule.
    """
    image = draw_molecule("CCO", input_format="smiles")

    assert image is not None


def test_draw_match_pair_returns_image():
    """
    draw_match_pair() should return an image object for a valid query-target pair.
    """
    image = draw_match_pair(
        "CC",
        "CCC",
        query_format="smiles",
        target_format="smiles",
        highlight=True,
    )

    assert image is not None


def test_to_networkx_if_available():
    """
    A SIGMo CSR graph should be convertible to a NetworkX graph when NetworkX is installed.
    """
    pytest.importorskip("networkx")

    graphs = sigmo.load_molecules(["CCO"], input_format="smiles")
    graph = graphs[0]

    G = to_networkx(graph)

    assert G.number_of_nodes() == 3
    assert G.number_of_edges() >= 2