import json
import math
from collections.abc import Mapping
from pathlib import Path

from pyproj import CRS
from pyproj.exceptions import CRSError

from .dependencies import require
from .errors import BackendExecutionError, DataValidationError, MissingBackendError

_NATIVE_EXTENSIONS = {".WT", ".WL", ".WP"}
_CRS_STATUSES = {"defined", "missing", "invalid", "unparsed"}


def _backend():
    return require("mapgis_annotations", "mapgis-annotations")


def _definition_crs(crs_id, definition):
    if not isinstance(definition, dict) or definition.get("status") != "defined":
        raise DataValidationError(f"Linked CRS definition is absent or not defined: {crs_id}")
    projjson, wkt = definition.get("projjson"), definition.get("wkt")
    if projjson is not None and not (isinstance(projjson, dict) and projjson):
        raise DataValidationError(f"CRS projjson is malformed for {crs_id}")
    if wkt is not None and not (isinstance(wkt, str) and wkt.strip()):
        raise DataValidationError(f"CRS WKT is malformed for {crs_id}")
    parsed_json = parsed_wkt = None
    if isinstance(projjson, dict) and projjson:
        try:
            parsed_json = CRS.from_user_input(projjson)
        except CRSError as exc:
            raise DataValidationError(f"CRS projjson cannot be parsed: {crs_id}") from exc
    if isinstance(wkt, str) and wkt.strip():
        try:
            parsed_wkt = CRS.from_user_input(wkt)
        except CRSError as exc:
            raise DataValidationError(f"CRS WKT cannot be parsed: {crs_id}") from exc
    if parsed_json is None and parsed_wkt is None:
        raise DataValidationError(f"Defined CRS lacks usable projjson or WKT: {crs_id}")
    if parsed_json is not None and parsed_wkt is not None and not parsed_json.equals(
            parsed_wkt, ignore_axis_order=True):
        raise DataValidationError(f"projjson and WKT disagree for CRS {crs_id}")
    crs = parsed_json if parsed_json is not None else parsed_wkt
    auth = definition.get("auth_id")
    if auth is not None:
        if not isinstance(auth, str) or ":" not in auth:
            raise DataValidationError(f"auth_id is malformed for CRS {crs_id}")
        try:
            CRS.from_user_input(auth)
        except CRSError as exc:
            raise DataValidationError(f"auth_id cannot be resolved for CRS {crs_id}") from exc
        identified_code, identified_auth = crs.to_epsg(), crs.to_authority(min_confidence=100)
        auth_pair = tuple(auth.split(":", 1))
        auth_code = int(auth_pair[1]) if auth_pair[0] == "EPSG" and auth_pair[1].isdigit() else None
        if (auth_code is not None and identified_code is not None and identified_code != auth_code) \
                or (identified_auth is not None and identified_auth != auth_pair):
            raise DataValidationError(f"auth_id does not match the CRS definition for {crs_id}")
    return crs


def _validate_reference(data):
    if not isinstance(data, dict):
        raise DataValidationError("Spatial reference must be a JSON object")
    fmt = data.get("format_version")
    if type(fmt) is not int or fmt != 1:
        raise DataValidationError(f"Unsupported spatial reference format_version: {fmt!r}")
    if data.get("kind") != "spatial_reference":
        raise DataValidationError("Not a spatial_reference report")
    if data.get("geometry_coordinate_order") != "xy":
        raise DataValidationError("Unsupported geometry coordinate order")
    layers, definitions = data.get("layers"), data.get("crs_definitions")
    if not isinstance(layers, dict) or not isinstance(definitions, dict):
        raise DataValidationError("layers and crs_definitions must be objects")
    for name, layer in layers.items():
        if not isinstance(layer, dict):
            raise DataValidationError(f"Layer entry is not an object: {name}")
        status = layer.get("crs_status")
        if not isinstance(status, str) or status not in _CRS_STATUSES:
            raise DataValidationError(f"Unknown crs_status for layer {name}: {status!r}")
        if status == "defined":
            crs_id = layer.get("crs_id")
            if not isinstance(crs_id, str) or not crs_id or crs_id not in definitions:
                raise DataValidationError(f"Layer crs_id is missing from crs_definitions: {name}")
            _definition_crs(crs_id, definitions[crs_id])
    return data


class MapGISAPI:
    def read_native(self, source, *, style=True):
        module = _backend()
        if not isinstance(style, bool):
            raise DataValidationError("style must be a boolean")
        path = Path(source)
        if not path.is_file() or path.suffix.upper() not in _NATIVE_EXTENSIONS:
            raise DataValidationError(f"source must be an existing MapGIS work file (.WT/.WL/.WP): {source}")
        try:
            return module.read_native_layer(path, style=style)
        except module.MissingDependencyError as exc:
            raise MissingBackendError(str(exc)) from exc
        except module.AnnotationError as exc:
            raise BackendExecutionError(str(exc)) from exc

    def export_native(self, source, output, *, epsg=None, factor=None, style=True, georeference=True):
        module = _backend()
        if not isinstance(style, bool) or not isinstance(georeference, bool):
            raise DataValidationError("style and georeference must be booleans")
        if epsg is not None and (isinstance(epsg, bool) or not isinstance(epsg, int) or epsg <= 0):
            raise DataValidationError("epsg must be a positive integer or None")
        if factor is not None and (isinstance(factor, bool) or not isinstance(factor, (int, float))
                                   or not math.isfinite(factor) or factor <= 0):
            raise DataValidationError("factor must be a finite positive number or None")
        output_path = Path(output)
        if output_path.exists():
            raise DataValidationError(f"output already exists: {output_path}")
        try:
            return module.export_native(source, output_path, epsg=epsg, factor=factor,
                                        style=style, georeference=georeference)
        except module.MissingDependencyError as exc:
            raise MissingBackendError(str(exc)) from exc
        except module.AnnotationError as exc:
            raise BackendExecutionError(str(exc)) from exc

    def read_reference(self, path):
        try:
            with open(path, "r", encoding="utf-8-sig") as stream:
                data = json.load(stream)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Spatial reference is not valid JSON: {path}") from exc
        return _validate_reference(data)

    def layer_crs(self, reference, layer):
        data = _validate_reference(reference) if isinstance(reference, Mapping) \
            else self.read_reference(reference)
        entry = data["layers"].get(layer)
        if entry is None:
            raise DataValidationError(f"Layer is absent from the spatial reference: {layer}")
        if entry["crs_status"] != "defined":
            raise DataValidationError(f"Layer CRS is '{entry['crs_status']}', not defined: {layer}")
        return _definition_crs(entry["crs_id"], data["crs_definitions"][entry["crs_id"]])