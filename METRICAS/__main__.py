"""
METRICAS.__main__
=================

Permite ejecutar la pipeline como::

    python -m METRICAS --input data.csv --output-dir out/
"""

from .pipeline import main

raise SystemExit(main())
