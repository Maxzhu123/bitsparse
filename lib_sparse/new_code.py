"""Compatibility entry points for the 15-bit implementation.

The implementation lives with the rest of the package kernels in
``src.code.bitpacking``.  These names preserve the prototype API used while
developing ``new_code.py``.
"""

try:
    from src.code.bitpacking import compress_15bit, uncompress_15bit
except ModuleNotFoundError:  # Imported as ``lib_sparse.new_code`` from repo root.
    from .src.code.bitpacking import compress_15bit, uncompress_15bit


def compress_fn(data, dtype=None, device=None):
    del dtype, device
    return compress_15bit(data)


def uncompress_fn(compressed_tensor, shape, dtype, device=None):
    del device
    return uncompress_15bit(compressed_tensor, shape, dtype)
