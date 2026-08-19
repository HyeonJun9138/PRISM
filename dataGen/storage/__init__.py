"""Dataset persistence layer.

Named ``storage`` and not ``io`` on purpose: ``io`` is a Python stdlib module
and shadowing it breaks unrelated imports.

Modules:
    schema   -- EpisodeArrays / EpisodeMeta, the on-disk contract + validator
    writer   -- (later phase) episode -> .npz + metadata on disk
    manifest -- (later phase) dataset-level index and split bookkeeping
"""
