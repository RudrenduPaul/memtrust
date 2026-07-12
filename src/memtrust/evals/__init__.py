"""Eval runners. Each module runs one eval family against any
MemoryBackendAdapter through the shared store()/query()/update()
interface, so scoring logic is written once and applied identically
across every tracked backend.
"""
