"""Analysis & presentation layer: episode JSONL -> paper tables, stats, and figures.

Kept separate from the run harness so it re-runs cheaply on partial or complete data.
Modules: ``aggregate`` (paired-aware outcome aggregation), ``stats``
(Wilson CI + paired/clustered bootstrap), ``link`` (forecast quality <-> outcome),
``transcript`` (readable example negotiations), ``figures`` (matplotlib artifacts).
"""
