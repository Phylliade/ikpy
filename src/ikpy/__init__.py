from ._version import __version__

# Check for JAX availability
try:
    import jax  # noqa: F401
    from . import jax_backend  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

__all__ = ["__version__", "JAX_AVAILABLE"]
