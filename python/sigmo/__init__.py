# Importo dal modulo compilato C++/pybind11 (_core)
# le funzioni che voglio rendere disponibili a livello Python.
#
# _core è il modulo nativo generato dalla build CMake:
# contiene il binding tra Python e le funzioni C++ implementate.
from ._core import (
    generate_csr_signatures,
    refine_csr_signatures,
    filter_candidates,
    Signature,
    Candidates,
    refine_candidates,
    join_candidates,
    GMCR,
)

# __all__ definisce l'interfaccia pubblica del package.
#
# In pratica dice: quando qualcuno usa from sigmo import *
# oppure quando vogliamo chiarire quali simboli sono "ufficialmente"
# esposti dal package, questi sono i nomi da considerare pubblici.
__all__ = [
    "generate_csr_signatures",
    "refine_csr_signatures",
    "filter_candidates",
    "Signature",
    "Candidates",
    "refine_candidates",
    "join_candidates",
    "GMCR",
]
