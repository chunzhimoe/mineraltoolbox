from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pyproj import CRS

from .dependencies import require
from .errors import DataValidationError


UNIT_TO_PPM = {"ppm": 1.0, "ppb": 0.001, "%": 10000.0, "wt%": 10000.0, "mg/kg": 1.0, "ug/g": 1.0, "µg/g": 1.0}


def record(operation: str, backend: str, parameters: Mapping[str, Any], **details: Any) -> dict:
    return {"operation": operation, "backend": backend, "parameters": dict(parameters), **details}


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (Path, CRS)):
        return str(value)
    raise TypeError(f"Cannot serialize metadata of type {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


@dataclass(frozen=True)
class GeochemSchema:
    sample_id: str
    x: str
    y: str
    crs: str | None
    elements: Sequence[str]
    units: Mapping[str, str]
    detection_limits: Mapping[str, float] = field(default_factory=dict)
    detection_limit_columns: Mapping[str, str] = field(default_factory=dict)
    zero_policy: str = "invalid"

    def __post_init__(self) -> None:
        elements = tuple(self.elements)
        if len(elements) < 1 or len(set(elements)) != len(elements):
            raise DataValidationError("Element names must be nonempty and unique")
        names = [self.sample_id, self.x, self.y, *elements]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise DataValidationError("ID, coordinate and element columns must have distinct nonempty names")
        if set(self.units) != set(elements):
            raise DataValidationError("Provide an explicit unit for every element, and no other columns")
        if any(unit not in UNIT_TO_PPM for unit in self.units.values()):
            raise DataValidationError(f"Supported units: {', '.join(UNIT_TO_PPM)}")
        if (set(self.detection_limits) | set(self.detection_limit_columns)) - set(elements):
            raise DataValidationError("Detection-limit configuration contains an unknown element")
        if set(self.detection_limits) & set(self.detection_limit_columns):
            raise DataValidationError("Choose a fixed limit OR a limit column for each element")
        if any(not np.isfinite(v) or v <= 0 for v in self.detection_limits.values()):
            raise DataValidationError("Detection limits must be finite and positive")
        if self.zero_policy not in {"invalid", "censored"}:
            raise DataValidationError("zero_policy must be 'invalid' or explicitly 'censored'")
        if self.crs is not None:
            CRS.from_user_input(self.crs)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "detection_limits", dict(self.detection_limits))
        object.__setattr__(self, "detection_limit_columns", dict(self.detection_limit_columns))


@dataclass
class GeochemDataset:
    raw: pd.DataFrame
    values: pd.DataFrame
    coordinates: pd.DataFrame
    status: pd.DataFrame
    detection_limits: pd.DataFrame
    schema: GeochemSchema
    crs: str | None
    domain: str = "concentration"
    units: dict[str, str] = field(default_factory=dict)
    imputed: pd.DataFrame | None = None
    history: list[dict] = field(default_factory=list)
    transform_state: dict | None = None

    def __post_init__(self) -> None:
        index = self.values.index
        if not index.is_unique or index.hasnans:
            raise DataValidationError("Dataset sample IDs must be unique and nonmissing")
        for table in (self.raw, self.coordinates, self.status, self.detection_limits):
            if not index.equals(table.index):
                raise DataValidationError("Dataset tables must use the same ordered sample IDs")
        if self.imputed is None:
            self.imputed = pd.DataFrame(False, index=index, columns=self.schema.elements)

    def copy(self, **changes: Any) -> GeochemDataset:
        import copy

        values = {name: copy.deepcopy(value) for name, value in vars(self).items()}
        values.update(changes)
        return GeochemDataset(**values)

    def export(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.raw.to_csv(directory / "raw.csv", index=False)
        for name in ("values", "coordinates", "status", "detection_limits", "imputed"):
            getattr(self, name).to_csv(directory / f"{name}.csv", index_label=self.schema.sample_id)
        write_json(directory / "metadata.json", {
            "domain": self.domain, "units": self.units, "crs": self.crs,
            "schema": vars(self.schema), "history": self.history,
            "transform_state": self.transform_state,
        })
        return directory


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]
    crs: str

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or isinstance(self.height, bool) or not isinstance(self.width, int) or not isinstance(self.height, int) or self.width <= 0 or self.height <= 0:
            raise DataValidationError("Grid dimensions must be positive integers")
        if len(self.transform) != 6 or not np.isfinite(self.transform).all():
            raise DataValidationError("Grid transform must contain six finite affine coefficients")
        a, b, _, d, e, _ = self.transform
        if a <= 0 or e >= 0 or b != 0 or d != 0:
            raise DataValidationError("First release supports north-up, unrotated grids only")
        CRS.from_user_input(self.crs)

    @classmethod
    def from_reference(cls, path: str | Path) -> GridSpec:
        rasterio = require("rasterio")
        with rasterio.open(path) as source:
            if source.crs is None:
                raise DataValidationError("Reference raster has no CRS")
            return cls(source.width, source.height, tuple(source.transform)[:6], source.crs.to_string())

    @classmethod
    def from_bounds(cls, bounds: Sequence[float], resolution: float, crs: str) -> GridSpec:
        if len(bounds) != 4 or not np.isfinite(bounds).all() or not np.isfinite(resolution) or resolution <= 0:
            raise DataValidationError("Provide finite bounds and a positive resolution")
        xmin, ymin, xmax, ymax = bounds
        if xmax <= xmin or ymax <= ymin:
            raise DataValidationError("Grid bounds must have positive area")
        nx, ny = (xmax - xmin) / resolution, (ymax - ymin) / resolution
        if not np.isclose(nx, round(nx), rtol=0, atol=1e-8) or not np.isclose(ny, round(ny), rtol=0, atol=1e-8):
            raise DataValidationError("Bounds must align exactly to the requested resolution")
        return cls(int(round(nx)), int(round(ny)), (resolution, 0.0, xmin, 0.0, -resolution, ymax), crs)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        a, _, c, _, e, f = self.transform
        return np.meshgrid(c + (np.arange(self.width) + 0.5) * a, f + (np.arange(self.height) + 0.5) * e)


@dataclass
class StepResult:
    data: GeochemDataset
    diagnostics: dict = field(default_factory=dict)


@dataclass
class MapResult:
    grid: GridSpec
    predictions: dict[str, np.ndarray]
    variances: dict[str, np.ndarray]
    support: np.ndarray
    domain: str
    units: dict[str, str]
    history: list[dict]

    def export(self, directory: str | Path) -> Path:
        rasterio = require("rasterio")
        affine = require("affine")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        profile = dict(driver="GTiff", width=self.grid.width, height=self.grid.height, count=1,
                       crs=self.grid.crs, transform=affine.Affine(*self.grid.transform), compress="deflate")
        names = list(self.predictions)
        layers = []
        for number, name in enumerate(names, 1):
            for suffix, arrays in (("prediction", self.predictions), ("variance", self.variances)):
                filename = f"{number:02d}_{suffix}.tif"
                with rasterio.open(directory / filename, "w", **profile, dtype="float64", nodata=np.nan) as output:
                    output.write(arrays[name].astype(np.float64), 1)
                    output.set_band_description(1, name)
                layers.append({"file": filename, "feature": name, "kind": suffix,
                               "unit": self.units[name] if suffix == "prediction" else f"({self.units[name]})^2"})
        with rasterio.open(directory / "support.tif", "w", **profile, dtype="uint8", nodata=0) as output:
            output.write(self.support.astype(np.uint8), 1)
        write_json(directory / "metadata.json", {
            "grid": vars(self.grid), "domain": self.domain, "units": self.units,
            "layers": layers, "history": self.history,
            "variance_meaning": "Conditional kriging variance in the output domain; not total exploration uncertainty",
        })
        return directory


@dataclass(frozen=True)
class GeochemJob:
    source: str | Path
    schema: GeochemSchema
    imputation: Mapping[str, Any] | None = None
    transformation: Mapping[str, Any] | None = None
    interpolation: Mapping[str, Any] | None = None
    read_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    cleaned_samples: GeochemDataset
    features: GeochemDataset | None
    maps: MapResult | None
    validation: dict
    history: list[dict]

    def export(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.cleaned_samples.export(directory / "samples")
        if self.features is not None:
            self.features.export(directory / "features")
        if self.maps is not None:
            self.maps.export(directory / "maps")
        write_json(directory / "run.json", {"validation": self.validation, "history": self.history})
        return directory
