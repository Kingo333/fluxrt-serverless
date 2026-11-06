"""Worker-side helpers for the FluxRT live try-on path (server.py).

Everything here is pure numpy/PIL so it can be unit-tested without the GPU
engine. MediaPipe is imported lazily inside the segmenter class only.
"""
