"""Postprocess helpers for prepared SHAR datasets.

Boundary:
- metadata-only scripts may rewrite cuts metadata while reusing existing
  recording tar shards
- any script that changes cut IDs, recording IDs, or shard membership must
  rebuild the matching recording tar shards too

Do not extend metadata-only patchers into ID-rewriting tools without also
upgrading them into full shard rewriters.
"""
