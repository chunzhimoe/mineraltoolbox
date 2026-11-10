import importlib.util
import shutil

from .dependencies import version
from .geochem import GeochemistryAPI
from .magnetic import MagneticAPI
from .mapgis import MapGISAPI

_BACKENDS = {
    "pyrolite": "pyrolite",
    "pykrige": "PyKrige",
    "gstools": "GSTools",
    "rasterio": "rasterio",
    "mapgis_annotations": "mapgis-annotations",
    "harmonica": "harmonica",
    "verde": "verde",
    "ppigrf": "ppigrf",
}


class GeoToolbox:
    def __init__(self):
        self.geochem = GeochemistryAPI()
        self.magnetic = MagneticAPI()
        self.mapgis = MapGISAPI()

    def backends(self):
        report = {}
        for module, distribution in _BACKENDS.items():
            report[module] = {
                "available": importlib.util.find_spec(module) is not None,
                "distribution": distribution,
                "version": version(distribution),
            }
        report["rscript"] = shutil.which("Rscript")
        return report