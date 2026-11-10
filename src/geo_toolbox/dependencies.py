import importlib
import importlib.metadata

from .errors import MissingBackendError


def require(module, distribution=None):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        top = module.split(".")[0]
        if exc.name == module or exc.name == top:
            dist = distribution or top
            raise MissingBackendError(
                f"Optional dependency '{module}' is not installed. "
                f"Install distribution '{dist}' (e.g. 'pip install {dist}') "
                "or the corresponding geo-toolbox extra."
            ) from exc
        raise


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"