from importlib import metadata


def get_version() -> str:
    """
    Return the installed version of the rag_app package if available.
    This is mainly for debugging and isn't required for core functionality.
    """
    try:
        return metadata.version("rag_app")
    except Exception:
        return "0.0.0"

