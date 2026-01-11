from ._version import __version__

__all__ = ["__version__"]

# Scarf analytics pixel - helps track library usage
def _scarf_analytics():
    """Download Scarf analytics pixel in background thread."""
    try:
        import sys
        from urllib.request import urlopen, Request
        from urllib.parse import urlencode
        
        params = urlencode({
            "x-pxid": "fad6aecc-1efc-4d90-a734-7f629e35c85b",
            "ikpy_version": __version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        })
        url = f"https://static.scarf.sh/a.png?{params}"
        
        req = Request(url, headers={"User-Agent": f"ikpy/{__version__}"})
        urlopen(req, timeout=2)
    except Exception:
        # Silently ignore any errors - analytics should never break the library
        pass

try:
    from threading import Thread
    Thread(target=_scarf_analytics, daemon=True).start()
except Exception:
    pass
