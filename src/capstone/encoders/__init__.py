"""Unified layer-wise extraction interface over all four encoders.

RQ3 requires identical treatment of WavJEPA, wav2vec 2.0, HuBERT, and WavLM:
same input handling, same per-layer hidden-state extraction, same output
shape conventions. Every encoder is wrapped here; probing code never imports
model libraries directly. Encoders are always frozen — nothing in this
package may expose a training path.
"""
