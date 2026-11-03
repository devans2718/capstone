"""Probe training and evaluation. Runs on cached embeddings only — never
touches an encoder or the GPU. Linear probes primary; the surface-vs-underlying
contrast is the core experimental design.
"""
