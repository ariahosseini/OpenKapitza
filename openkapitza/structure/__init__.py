"""Phase 2: structure building and validation.

`analysis` measures (density, RDF, coordination, angles); `crystals` builds
ideal cells; `amorphous` makes and validates a melt-quenched glass;
`interface` joins a crystal to a glass for NEMD.
"""

from . import amorphous, analysis, crystalline, crystals, interface  # noqa: F401
