"""
SIGMo Python interface.

The public API is organized into three layers:

1. High-level API:
   match(), search(), run_isomorphism(), load_molecules()

2. Mid-level API:
   SIGMoMatcher and PipelineContext for users who need more control over
   allocation, filtering, refinement and join steps.

3. Low-level API:
   native C++/SYCL kernels and data structures exposed through pybind11.

Notes:
    The package imports dpctl before loading the native extension to avoid
    SYCL/Unified Runtime library loading conflicts in some Conda/oneAPI setups.

    Visualize is not exported here because it has dependeces (networkx, matplotlib). 
    To make the interfeace more stable, I chose to keep it out from the top-level.
"""

# Import dpctl before loading the native SIGMo extension.
# This avoids SYCL/Unified Runtime library loading conflicts in some environments.
try:
    import dpctl as _dpctl  
except Exception:
    # The actual error will be raised later if a SYCL queue is required.
    pass

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

# Low-level API: available for advanced/HPC users.
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