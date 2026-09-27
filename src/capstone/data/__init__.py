"""Corpus download, preparation, and speaker-disjoint splits.

Split manifests are written to the repo (not /data) and committed: if they
were regenerated differently, every cached embedding's split assignment
would silently break.
"""
