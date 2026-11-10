from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError

from .core import GridSpec, record, write_json
from .dependencies import require, version
from .errors import DataValidationError

MAGNETIC_STAGES = frozenset({"raw_total_field", "corrected_total_field", "anomaly", "rtp"})
_MAGNETIC_UNITS = {"nT": 1.0, "uT": 1e3, "T": 1e9}
_GRID_UNITS = {"nT", "nT/m", "nT/m^2", "rad"}
_COORDINATE_COLUMNS = ["easting", "northing", "upward"]
_EXCEL_ENGINES = {".xls": "xlrd", ".xlsx": "openpyxl", ".xlsm": "openpyxl"}


@dataclass(frozen=True)
class MagneticSchema:
    sample_id: str
    x: str
    y: str
    value: str
    crs: str | None
    unit: str
    stage: str
    upward: str | None = None
    line: str | None = None
    sequence: str | None = None

    def __post_init__(self) -> None:
        required = [self.sample_id, self.x, self.y, self.value]
        if any(not isinstance(column, str) or not column.strip() for column in required):
            raise DataValidationError("ID, coordinate and value columns must be nonempty strings")
        optional = [self.upward, self.line, self.sequence]
        if any(column is not None and (not isinstance(column, str) or not column.strip())
               for column in optional):
            raise DataValidationError("Configured optional columns must be nonempty strings")
        configured = required + [column for column in optional if column is not None]
        if len(set(configured)) != len(configured):
            raise DataValidationError("ID, coordinate, value and optional columns must have distinct names")
        if not isinstance(self.unit, str) or self.unit not in _MAGNETIC_UNITS:
            raise DataValidationError(f"Supported magnetic units: {', '.join(_MAGNETIC_UNITS)}")
        if not isinstance(self.stage, str) or self.stage not in MAGNETIC_STAGES:
            raise DataValidationError(f"Supported magnetic stages: {', '.join(sorted(MAGNETIC_STAGES))}")
        if self.crs is not None:
            if not isinstance(self.crs, str):
                raise DataValidationError("crs must be a string or None")
            try:
                CRS.from_user_input(self.crs)
            except CRSError as exc:
                raise DataValidationError(f"Invalid CRS: {self.crs}") from exc


@dataclass
class MagneticDataset:
    raw: pd.DataFrame
    values: pd.Series
    coordinates: pd.DataFrame
    status: pd.Series
    schema: MagneticSchema
    stage: str
    active: pd.Series | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or self.stage not in MAGNETIC_STAGES:
            raise DataValidationError(f"Supported magnetic stages: {', '.join(sorted(MAGNETIC_STAGES))}")
        index = self.values.index
        if not index.is_unique or index.hasnans \
                or any(value is None or (isinstance(value, str) and not value.strip()) for value in index):
            raise DataValidationError("Magnetic sample IDs must be unique and nonmissing")
        for table in (self.raw, self.coordinates, self.status):
            if not index.equals(table.index):
                raise DataValidationError("Magnetic tables must use the same ordered sample IDs")
        if list(self.coordinates.columns) != _COORDINATE_COLUMNS:
            raise DataValidationError("Coordinates must use the canonical easting/northing/upward columns")
        if self.active is None:
            self.active = pd.Series(True, index=index, name="active")
        elif not index.equals(self.active.index) or self.active.dtype != bool:
            raise DataValidationError("Active flags must be a boolean column on the same sample IDs")

    def copy(self, **changes: Any) -> "MagneticDataset":
        values = {name: copy.deepcopy(value) for name, value in vars(self).items()}
        values.update(changes)
        return MagneticDataset(**values)

    def export(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.raw.to_csv(directory / "raw.csv", index=False)
        for name in ("values", "coordinates", "status", "active"):
            getattr(self, name).to_csv(directory / f"{name}.csv", index_label=self.schema.sample_id)
        write_json(directory / "metadata.json", {
            "schema": vars(self.schema), "stage": self.stage, "unit": "nT",
            "crs": self.schema.crs, "history": self.history,
        })
        return directory


@dataclass
class MagneticGrid:
    grid: GridSpec
    values: np.ndarray
    support: np.ndarray
    stage: str
    unit: str
    upward: float | None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or self.stage not in MAGNETIC_STAGES:
            raise DataValidationError(f"Supported magnetic stages: {', '.join(sorted(MAGNETIC_STAGES))}")
        if not isinstance(self.unit, str) or self.unit not in _GRID_UNITS:
            raise DataValidationError(f"Supported magnetic grid units: {', '.join(sorted(_GRID_UNITS))}")
        shape = (self.grid.height, self.grid.width)
        values = np.asarray(self.values)
        if values.shape != shape or values.dtype.kind not in "fiu":
            raise DataValidationError("Grid values must be a numeric array matching the grid shape")
        if np.isinf(values).any():
            raise DataValidationError("Grid values cannot contain infinities")
        self.values = np.array(values, dtype=np.float64, copy=True)
        support = np.asarray(self.support)
        if support.shape != shape or support.dtype != bool:
            raise DataValidationError("Support must be a boolean array matching the grid shape")
        self.support = np.array(support, dtype=bool, copy=True)
        if self.upward is not None:
            if isinstance(self.upward, bool) or not isinstance(self.upward, (int, float)) \
                    or not math.isfinite(self.upward):
                raise DataValidationError("upward must be a finite scalar or None")
            self.upward = float(self.upward)

    def copy(self, **changes: Any) -> "MagneticGrid":
        values = {name: copy.deepcopy(value) for name, value in vars(self).items()}
        values.update(changes)
        return MagneticGrid(**values)

    def export(self, directory: str | Path) -> Path:
        rasterio = require("rasterio")
        affine = require("affine")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        profile = dict(driver="GTiff", width=self.grid.width, height=self.grid.height, count=1,
                       crs=self.grid.crs, transform=affine.Affine(*self.grid.transform), compress="deflate")
        with rasterio.open(directory / "values.tif", "w", **profile, dtype="float64", nodata=np.nan) as output:
            output.write(np.where(self.support, self.values, np.nan).astype(np.float64), 1)
        with rasterio.open(directory / "support.tif", "w", **profile, dtype="uint8", nodata=0) as output:
            output.write(self.support.astype(np.uint8), 1)
        write_json(directory / "metadata.json", {
            "grid": vars(self.grid), "stage": self.stage, "unit": self.unit, "upward": self.upward,
            "history": self.history,
            "support_meaning": "sampling/domain mask; not statistical uncertainty",
        })
        return directory


@dataclass
class MagneticQC:
    data: MagneticDataset
    diagnostics: dict


def read_magnetic(source: str | Path | pd.DataFrame, *, schema: MagneticSchema,
                  sheet_name: Any = 0, header: Any = 0, skiprows: Any = None) -> MagneticDataset:
    if not isinstance(schema, MagneticSchema):
        raise DataValidationError("schema must be a MagneticSchema")
    frame = source.copy(deep=True) if isinstance(source, pd.DataFrame) else None
    if frame is None:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            if isinstance(header, (list, tuple)):
                raise DataValidationError("Multirow headers are currently supported for Excel inputs only")
            frame = pd.read_csv(path, dtype={schema.sample_id: "string"}, keep_default_na=False,
                                header=header, skiprows=skiprows)
        elif suffix in _EXCEL_ENGINES:
            if sheet_name is None:
                raise DataValidationError("sheet_name=None would load every sheet; choose one explicitly")
            engine = _EXCEL_ENGINES[suffix]
            require(engine)
            frame = pd.read_excel(path, dtype=str, keep_default_na=False, sheet_name=sheet_name,
                                  header=header, skiprows=skiprows, engine=engine)
            if isinstance(frame, dict):
                raise DataValidationError("sheet_name must resolve to a single sheet")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = [next((str(part).strip() for part in reversed(parts)
                                       if not str(part).startswith("Unnamed:") and str(part).strip()), "")
                                 for parts in frame.columns]
        else:
            raise DataValidationError(f"Unsupported magnetic source: {path}")
    if frame.empty or frame.columns.hasnans \
            or any(not isinstance(name, str) or not name.strip() for name in frame.columns) \
            or frame.columns.duplicated().any():
        raise DataValidationError("Source columns must be nonempty and unique")
    required = {schema.sample_id, schema.x, schema.y, schema.value}
    required |= {column for column in (schema.upward, schema.line, schema.sequence) if column is not None}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")
    ids = frame[schema.sample_id]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise DataValidationError("Sample IDs must be nonmissing, unique and nonblank")
    frame = frame.set_index(schema.sample_id, drop=False)
    factor = _MAGNETIC_UNITS[schema.unit]
    tokens = frame[schema.value]
    parsed = pd.to_numeric(tokens, errors="coerce")
    empty = tokens.isna() | (tokens.astype(str).str.strip() == "")
    finite = parsed.notna() & np.isfinite(parsed)
    status = pd.Series(np.select([empty, ~empty & finite], ["missing", "observed"], default="invalid"),
                       index=frame.index, name="status")
    values = (parsed.astype(np.float64) * factor).rename(schema.value)
    easting = pd.to_numeric(frame[schema.x], errors="coerce")
    northing = pd.to_numeric(frame[schema.y], errors="coerce")
    upward = pd.to_numeric(frame[schema.upward], errors="coerce") if schema.upward is not None \
        else pd.Series(np.nan, index=frame.index)
    coordinates = pd.DataFrame({"easting": easting, "northing": northing, "upward": upward})
    active = pd.Series(True, index=frame.index, name="active")
    entry = record("read", "pandas", {
        "source": "dataframe" if isinstance(source, pd.DataFrame) else str(source),
        "sheet_name": sheet_name, "header": header, "skiprows": skiprows,
        "original_unit": schema.unit, "output_unit": "nT", "stage": schema.stage,
    }, version=version("pandas"))
    return MagneticDataset(raw=frame, values=values, coordinates=coordinates, status=status,
                           schema=schema, stage=schema.stage, active=active, history=[entry])