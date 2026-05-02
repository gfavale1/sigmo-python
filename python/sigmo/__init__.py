"""
SIGMo Python interface.

La API pubblica e' organizzata su tre livelli:
1. high-level: match(), search(), load_molecules()
2. mid-level: SIGMoMatcher, PipelineContext
3. low-level: kernel C++ esposti dal binding pybind11
"""

from .matcher import SIGMoMatcher, match, match_smarts, run_isomorphism, search
from .graph import (
    chemical_string_to_csr,
    from_networkx,
    load_molecules,
    make_csr_graph,
    rdkit_mol_to_csr,
    smarts_to_csr,
    smarts_to_csr_from_string,
    to_networkx,
    toy_two_node_graph,
)
from .pipeline import PipelineContext
from .result import KernelStep, Match, MatchResult
from .config import get_default_queue, get_sycl_queue
from .validation import validate_result_with_rdkit

# Low-level API: resta disponibile per utenti avanzati/HPC.
from ._core import (
    Candidates,
    GMCR,
    Signature,
    filter_candidates,
    generate_csr_signatures,
    join_candidates,
    refine_candidates,
    refine_csr_signatures,
)

__all__ = [
    # High-level API
    "match",
    "match_smarts",
    "search",
    "run_isomorphism",
    "SIGMoMatcher",
    "load_molecules",
    # Graph utilities
    "make_csr_graph",
    "chemical_string_to_csr",
    "smarts_to_csr_from_string",
    "smarts_to_csr",
    "rdkit_mol_to_csr",
    "toy_two_node_graph",
    "to_networkx",
    "from_networkx",
    # Pipeline/result
    "PipelineContext",
    "MatchResult",
    "Match",
    "KernelStep",
    "validate_result_with_rdkit",
    # Config
    "get_sycl_queue",
    "get_default_queue",
    # Low-level kernels/classes
    "generate_csr_signatures",
    "refine_csr_signatures",
    "filter_candidates",
    "refine_candidates",
    "join_candidates",
    "Signature",
    "Candidates",
    "GMCR",
]
