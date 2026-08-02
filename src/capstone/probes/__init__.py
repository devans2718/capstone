"""Probe training and evaluation. Runs on cached embeddings only — never
touches an encoder or the GPU. Linear probes primary; see the vault's probe
design principles for the surface-vs-underlying contrast that is the core
experimental design.
"""
