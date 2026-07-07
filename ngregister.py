import argparse
import webbrowser

import neuroglancer
import neuroglancer.cli
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import fftconvolve
import datetime
import atexit

import json
import SimpleITK as sitk
import tensorstore as ts

viewer = neuroglancer.Viewer()

angle_step_in_degrees = 3

history = {}

# Helper function for rotation
def compute_rotation_around_main_axis(s,axis_name):
    global angle_step_in_degrees
    sign = 1
    if "-" in axis_name:
        axis_name = axis_name.replace("-","")
        sign = -1
    if axis_name not in ["x","y","z"] or len(axis_name) != 1:
        raise Exception("axis_name must be standard x, y or z, eventually with a \"-\" sign")
    center_to_origin = - np.array(s.viewer_state.voxel_coordinates).reshape((3,1))
    rot = Rotation.from_euler(axis_name,sign*angle_step_in_degrees,degrees=True)

    return np.block([
            [np.eye(3), - center_to_origin],
            [np.zeros((1,3)), np.ones(1)]
        ]) @ np.block([
            [rot.as_matrix(), np.zeros((3,1))],
            [np.zeros((1,3)), np.ones(1)]
        ]) @ np.block([
            [np.eye(3), center_to_origin],
            [np.zeros((1,3)), np.ones(1)]
        ])

def flip_main_axis(s,axis_name):
    if axis_name not in ["x","y","z"] or len(axis_name) != 1:
        raise Exception("axis_name must be standard x, y or z")
    axis_id = {"x":0,"y":1,"z":2}[axis_name]
    center_to_origin = - np.array(s.viewer_state.voxel_coordinates).reshape((3,1))
    rot = Rotation.identity().as_matrix()
    rot[axis_id,axis_id] = -1

    return np.block([
            [np.eye(3), - center_to_origin],
            [np.zeros((1,3)), np.ones(1)]
        ]) @ np.block([
            [rot, np.zeros((3,1))],
            [np.zeros((1,3)), np.ones(1)]
        ]) @ np.block([
            [np.eye(3), center_to_origin],
            [np.zeros((1,3)), np.ones(1)]
        ])

# Helper function for all transforms
def apply_transform_to_layer(applied_transform_matrix, chain_backwards: bool = False, layer_name=None):
    global viewer
    with viewer.txn() as v:
        if layer_name is None:
            layer_name = v.selectedLayer.layer
        if v.layers[layer_name].layer.type == "annotation":
            return
        current_transform = v.layers[layer_name].layer.source[0].transform
        if current_transform == None:
            current_transform_matrix = np.eye(4)
        else:
            current_transform_matrix = np.eye(4)
            current_transform_matrix[:3,:4] = np.array(current_transform.matrix)
        if chain_backwards:
            new_transform_matrix = current_transform_matrix @ applied_transform_matrix
        else:
            new_transform_matrix = applied_transform_matrix @ current_transform_matrix
        new_transform = neuroglancer.CoordinateSpaceTransform({"matrix": new_transform_matrix[:3,:4].tolist(), "outputDimensions": v.dimensions.to_json()})
        current_source_url = v.layers[layer_name].layer.source[0].url
        v.layers[layer_name].layer.source[0] = neuroglancer.LayerDataSource({"url": current_source_url, "transform": new_transform.to_json()})

def save_state_in_history():
    global history
    global viewer
    timestamp = datetime.datetime.now().isoformat()
    with viewer.txn() as v:
        history["history_"+timestamp] = v.to_json()
        if "__HISTORY__" not in v.layers:
            v.layers.append(
                name="__HISTORY__",
                layer=neuroglancer.LocalAnnotationLayer(
                    dimensions=v.dimensions,
                    annotationColor= "#ff0000",
                ),
            )
        v.layers["__HISTORY__"].annotations.append(
            neuroglancer.PointAnnotation(
                id="history_"+timestamp,
                point=[0,0,0],
                description=timestamp
            )
        )

def load_state_from_history():
    global history
    global viewer
    with viewer.txn() as v:
        if "__HISTORY__" not in v.layers:
            return
        if v.selectedLayer.layer == "__HISTORY__" and v.selection.layers.annotation.annotationId is not None:
            v = history[v.selection.layers.annotation.annotationId]
        else:
            v = history[sorted(history.keys())[-1]]

# Rotation actions
def rotate_layer_absolute_z_clockwise(s):
    axis = "z"
    apply_transform_to_layer(
        compute_rotation_around_main_axis(s,axis)
    )
    save_state_in_history()
def rotate_layer_absolute_y_clockwise(s):
    axis = "-y"
    apply_transform_to_layer(
        compute_rotation_around_main_axis(s,axis)
    )
    save_state_in_history()
def rotate_layer_absolute_x_clockwise(s):
    axis = "x"
    apply_transform_to_layer(
        compute_rotation_around_main_axis(s,axis)
    )
    save_state_in_history()
def rotate_layer_absolute_z_counterclockwise(s):
    axis = "-z"
    apply_transform_to_layer(
        compute_rotation_around_main_axis(s,axis)
    )
    save_state_in_history()
def rotate_layer_absolute_y_counterclockwise(s):
    axis = "y"
    apply_transform_to_layer(
        compute_rotation_around_main_axis(s,axis)
    )
    save_state_in_history()
def rotate_layer_absolute_x_counterclockwise(s):
    axis = "-x"
    apply_transform_to_layer(
        compute_rotation_around_main_axis(s,axis)
    )
    save_state_in_history()

# Flipping actions
def flip_x_axis(s):
    axis = "x"
    apply_transform_to_layer(
        flip_main_axis(s,axis)
    )
    save_state_in_history()
def flip_y_axis(s):
    axis = "y"
    apply_transform_to_layer(
        flip_main_axis(s,axis)
    )
def flip_z_axis(s):
    axis = "z"
    apply_transform_to_layer(
        flip_main_axis(s,axis)
    )

# Elementary translation actions
# NOT USED ANYMORE
# def translate_layer_absolute_x(s):
#     apply_transform_to_layer(
#         np.block([
#             [np.eye(3), np.asmatrix("10;0;0")],
#             [np.zeros((1,3)), np.ones(1)]
#         ])
#     )

# def translate_layer_absolute_y(s):
#     apply_transform_to_layer(
#         np.block([
#             [np.eye(3), np.asmatrix("0;10;0")],
#             [np.zeros((1,3)), np.ones(1)]
#         ])
#     )

# def translate_layer_absolute_z(s):
#     apply_transform_to_layer(
#         np.block([
#             [np.eye(3), np.asmatrix("0;0;10")],
#             [np.zeros((1,3)), np.ones(1)]
#         ])
#     )

# Complex translation actions
def translate_layer_from_cursor_to_center(s):
    if s.mouse_voxel_coordinates is None:
        return
    translation_vector = s.viewer_state.voxel_coordinates - s.mouse_voxel_coordinates
    homogeneous_matrix = np.block([
            [np.eye(3), translation_vector.reshape(3, 1)],
            [np.zeros((1,3)), np.ones(1)]
            ])
    apply_transform_to_layer(
        homogeneous_matrix
    )

def place_translation_landmark(s):
    global viewer
    if s.mouse_voxel_coordinates is None:
        return
    with viewer.txn() as v:
        if "__LANDMARK__" not in v.layers:
            v.layers.append(
                name="__LANDMARK__",
                layer=neuroglancer.LocalAnnotationLayer(
                    dimensions=v.dimensions,
                    annotationColor= "#ff0000",
                ),
            )
        landmark_position = neuroglancer.PointAnnotation(
                    id=neuroglancer.random_token.make_random_token(),
                    point=s.mouse_voxel_coordinates,
                )
        if len(v.layers["__LANDMARK__"].annotations) == 0:
            v.layers["__LANDMARK__"].annotations.append(
                landmark_position
            )
        else:
            v.layers["__LANDMARK__"].annotations[0] = landmark_position

def translate_layer_to_landmark(s):
    global viewer
    if s.mouse_voxel_coordinates is None:
        return
    if "__LANDMARK__" in s.viewer_state.layers and len(s.viewer_state.layers["__LANDMARK__"].annotations) == 1:
        landmark_position = s.viewer_state.layers["__LANDMARK__"].annotations[0].point
        translation_vector = landmark_position - s.mouse_voxel_coordinates
        homogeneous_matrix = np.block([
                [np.eye(3), translation_vector.reshape(3, 1)],
                [np.zeros((1,3)), np.ones(1)]
                ])
        apply_transform_to_layer(
            homogeneous_matrix
        )
        with viewer.txn() as v:
            v.voxel_coordinates = landmark_position

# ---------------------------------------------------------------------------
# Registration refinement (TensorStore + SimpleITK)
#
# refine_registration() fetches a small subvolume around a landmark from a
# reference and a moving layer, runs an affine SimpleITK registration, and
# composes the correction onto the moving layer. The three libraries involved
# use different axis conventions, so everything is expressed through explicit
# affine bookkeeping in a single shared frame (global physical space):
#
# - Neuroglancer: positions (voxel_coordinates, mouse_voxel_coordinates,
#   annotation points) are in global/output voxel coordinates, ordered like
#   viewer.dimensions. A layer's source[0].transform.matrix is a 3x4 affine
#   mapping local voxel -> global voxel.
# - TensorStore: opens the source lazily (precomputed, zarr, zarr3 incl.
#   sharding, n5) and only the small cutout is read into memory. The returned
#   array's axes follow the source's stored order, which is also the column
#   order of the layer transform; dimension labels identify the channel axis to
#   drop and confirm the spatial axes.
# - SimpleITK: an Image is indexed (x, y, z), but GetImageFromArray /
#   GetArrayFromImage use the reversed numpy order [z, y, x]. Geometry lives in
#   physical space via origin / spacing / direction.

MOVING_PREFIX = "mov::"
REFERENCE_PREFIX = "ref::"

# Layer role tagging
def _strip_role_prefix(name):
    for prefix in (MOVING_PREFIX, REFERENCE_PREFIX):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def mark_active_layer(prefix, name=None):
    global viewer
    with viewer.txn() as v:
        if name is None:
            name = v.selectedLayer.layer
        if name is None or name not in v.layers:
            print(f"mark: no layer to tag (name={name!r}); select a layer in the "
                  "viewer or pass name='<layer>'.")
            return
        new_name = prefix + _strip_role_prefix(name)
        v.layers[name].name = new_name
        print(f"mark: '{name}' -> '{new_name}'")

def mark_layer_moving(s=None):
    mark_active_layer(MOVING_PREFIX)

def mark_layer_reference(s=None):
    mark_active_layer(REFERENCE_PREFIX)

# REPL-callable equivalents of the alt+m / alt+r keybindings, for when the
# browser swallows the Alt+key shortcut. `name` defaults to the selected layer.
def mark_moving(name=None):
    mark_active_layer(MOVING_PREFIX, name)

def mark_reference(name=None):
    mark_active_layer(REFERENCE_PREFIX, name)

# Neuroglancer data-source format -> TensorStore driver.
_DRIVER_BY_FORMAT = {
    "precomputed": "neuroglancer_precomputed",
    "zarr": "zarr",
    "zarr2": "zarr",
    "zarr3": "zarr3",
    "n5": "n5",
}
# TensorStore kvstore URL schemes that can be passed through unchanged.
_KVSTORE_SCHEMES = ("gs://", "s3://", "http://", "https://", "file://", "memory://")
# Dimension-label names that are not spatial and are reduced to their first index.
_NON_SPATIAL_LABELS = {"channel", "c", "t", "time"}
# Physical length units -> metres, for comparing source and viewer scales.
_UNIT_TO_METERS = {
    "m": 1.0, "meter": 1.0, "metre": 1.0,
    "cm": 1e-2, "centimeter": 1e-2, "centimetre": 1e-2,
    "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
    "um": 1e-6, "µm": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6, "micron": 1e-6,
    "nm": 1e-9, "nanometer": 1e-9, "nanometre": 1e-9,
    "angstrom": 1e-10, "Å": 1e-10,
}


def _unit_to_meters(unit):
    """Length unit string -> metres; None/empty/unknown -> 1.0 (treat as ratio)."""
    if not unit:
        return 1.0
    return _UNIT_TO_METERS.get(str(unit).strip().lower(), 1.0)


def split_source_url(url):
    """Split a Neuroglancer source URL into a (format, kvstore) pair.

    Handles both encodings: the legacy ``<format>://<kvstore>`` (e.g.
    ``precomputed://gs://bucket/path``) and the newer ``<kvstore>|<format>:``
    suffix (e.g. ``https://host/data.ome.zarr|zarr3:``).
    """
    if "|" in url:
        kvstore, fmt = url.rsplit("|", 1)
        fmt = fmt.rstrip(":")
    else:
        fmt, kvstore = url.split("://", 1)
    if not kvstore.startswith(_KVSTORE_SCHEMES):
        # A bare path is a local filesystem store.
        kvstore = "file://" + kvstore
    return fmt.lower(), kvstore


def _read_json_key(kvstore, key):
    """Read and parse a JSON metadata key from a TensorStore kvstore, or None."""
    try:
        store = ts.KvStore.open(kvstore).result()
        result = store.read(key).result()
    except Exception:
        return None
    if not result.value:
        return None
    return json.loads(bytes(result.value))


def _ome_multiscale(driver, kvstore):
    """Return the first OME-NGFF ``multiscales`` entry for a zarr group, or None.

    Reads the group metadata (``zarr.json`` for v3, ``.zattrs`` for v2). Returns
    None when ``kvstore`` is not an OME multiscale group (e.g. a plain array).
    """
    multiscales = None
    if driver == "zarr3":
        meta = _read_json_key(kvstore, "zarr.json")
        if meta and meta.get("node_type") == "group":
            attributes = meta.get("attributes", {})
            ome = attributes.get("ome", attributes)
            multiscales = ome.get("multiscales")
    elif driver == "zarr":
        multiscales = (_read_json_key(kvstore, ".zattrs") or {}).get("multiscales")
    else:
        multiscales = None
    return multiscales[0] if multiscales else None


def _resolve_multiscale_kvstore(driver, kvstore, mip):
    """If ``kvstore`` points at an OME multiscale group, descend to level ``mip``.

    TensorStore opens a single array, not an OME-NGFF multiscale group, so the
    kvstore path must point at the desired resolution array. Reads the group
    metadata to find the dataset path. Returns ``kvstore`` unchanged when it is
    not a multiscale group.
    """
    multiscale = _ome_multiscale(driver, kvstore)
    if multiscale:
        datasets = multiscale["datasets"]
        return kvstore.rstrip("/") + "/" + datasets[int(mip)]["path"]
    return kvstore


def _native_voxel_geometry(url, mip, spatial, global_scale_phys):
    """Per-spatial-axis ``(ratio, offset)`` mapping array index -> intrinsic coord.

    The neuroglancer source ``transform.matrix`` does not map the raw array
    index to global voxels: it maps the source's *physical* position expressed
    in global-voxel units (the source's intrinsic coordinate). This converts an
    array index to that intrinsic coordinate per axis::

        intrinsic = ratio * index + offset

    with ``ratio = native_voxel_size / global_voxel_size`` and
    ``offset = native_origin / global_voxel_size`` (both in metres / metres, so
    unitless). Native geometry comes from the OME ``coordinateTransformations``
    (zarr) or the precomputed ``resolution`` (precomputed). Falls back to
    ``(1, 0)`` -- index == intrinsic -- when no resolution metadata is found,
    which preserves behaviour for sources stored at the global resolution.

    ``global_scale_phys`` is the viewer dimensions' physical voxel size in metres
    (one entry per spatial axis, in the transform's column order).
    """
    fmt, kvstore = split_source_url(url)
    driver = _DRIVER_BY_FORMAT.get(fmt)
    native_scale_m = None
    native_trans_m = None

    if driver in ("zarr", "zarr3"):
        multiscale = _ome_multiscale(driver, kvstore)
        if multiscale:
            axes = multiscale.get("axes", [])
            datasets = multiscale["datasets"]
            transforms = datasets[min(int(mip), len(datasets) - 1)].get(
                "coordinateTransformations", [])
            scale = next((t["scale"] for t in transforms
                          if t.get("type") == "scale"), None)
            translation = next((t["translation"] for t in transforms
                                if t.get("type") == "translation"), None)
            if scale is not None:
                factors = [_unit_to_meters(axes[a].get("unit")) if a < len(axes)
                           else 1.0 for a in spatial]
                native_scale_m = [scale[a] * f for a, f in zip(spatial, factors)]
                if translation is not None:
                    native_trans_m = [translation[a] * f
                                      for a, f in zip(spatial, factors)]
    elif driver == "neuroglancer_precomputed":
        info = _read_json_key(kvstore, "info")
        scales = (info or {}).get("scales")
        if scales:
            resolution = scales[min(int(mip), len(scales) - 1)].get("resolution")
            if resolution is not None:
                # precomputed resolutions are in nanometres
                native_scale_m = [resolution[a] * 1e-9 for a in spatial]

    if native_scale_m is None:
        return np.ones(len(spatial)), np.zeros(len(spatial))

    global_scale_phys = np.asarray(global_scale_phys, dtype=float)
    ratio = np.asarray(native_scale_m, dtype=float) / global_scale_phys
    if native_trans_m is None:
        offset = np.zeros(len(spatial))
    else:
        offset = np.asarray(native_trans_m, dtype=float) / global_scale_phys
    return ratio, offset


def source_url_to_spec(url, mip=0):
    """Turn a Neuroglancer source URL into a TensorStore open spec.

    Pure (no I/O): selects the driver from the source format and sets
    ``scale_index`` for precomputed. OME multiscale resolution for zarr is
    deferred to ``_open_source`` so spec building never does (hangable) network
    reads.
    """
    fmt, kvstore = split_source_url(url)
    driver = _DRIVER_BY_FORMAT.get(fmt)
    if driver is None:
        raise RuntimeError(f"refine: unsupported source format '{fmt}' in '{url}'")
    spec = {"driver": driver, "kvstore": kvstore}
    if driver == "neuroglancer_precomputed":
        spec["scale_index"] = int(mip)
    return spec


def _open_source(url, mip):
    """Lazily open a Neuroglancer source URL as a TensorStore dataset.

    If a zarr source is an OME multiscale *group* rather than a single array,
    the first open fails; we then resolve the level-``mip`` dataset path from
    the group metadata and retry once.
    """
    spec = source_url_to_spec(url, mip)
    try:
        return ts.open(spec).result()
    except Exception as exc:
        if spec["driver"] in ("zarr", "zarr3"):
            resolved = _resolve_multiscale_kvstore(spec["driver"], spec["kvstore"], mip)
            if resolved != spec["kvstore"]:
                return ts.open({**spec, "kvstore": resolved}).result()
        raise RuntimeError(
            f"refine: TensorStore could not open source '{url}': {exc}"
        ) from exc


def _spatial_axis_order(labels, rank, ndim):
    """Return (spatial_axes, nonspatial_axes) for a TensorStore dataset.

    Spatial axes are kept in their native storage order, which matches the
    column order of the layer transform. Channel/time axes are reduced to their
    first index. Falls back to the trailing ``ndim`` axes when labels are absent.
    """
    named = [i for i, label in enumerate(labels)
             if label and label.lower() not in _NON_SPATIAL_LABELS]
    if len(named) == ndim:
        spatial = named
    elif rank == ndim:
        spatial = list(range(rank))
    else:
        spatial = list(range(rank - ndim, rank))
        print("refine: warning - unlabeled source dimensions; assuming the "
              f"trailing {ndim} axes are spatial.")
    nonspatial = [i for i in range(rank) if i not in spatial]
    return spatial, nonspatial

def layer_transform_matrix(layer, ndim):
    """Return the 4x4 local-voxel -> global-voxel affine for a layer source.

    Mirrors the `current_transform == None` handling in
    apply_transform_to_layer: a missing transform is the identity.
    """
    transform = layer.source[0].transform
    matrix = np.eye(ndim + 1)
    if transform is not None:
        matrix[:ndim, : ndim + 1] = np.array(transform.matrix)
    return matrix

def polar_decompose(linear):
    """Decompose a linear map into U @ diag(spacing).

    U is orthonormal (used as the SimpleITK image direction) and spacing is the
    per-axis scale. This is exact when the input is a rotation composed with
    anisotropic scaling (no shear), which is what ngregister's rigid + flip
    gestures produce. Shear is detected and reported.
    """
    w, s, vt = np.linalg.svd(linear)
    u = w @ vt
    p = vt.T @ np.diag(s) @ vt
    spacing = np.diag(p).copy()
    off_diagonal = np.abs(p - np.diag(spacing)).max()
    if off_diagonal > 1e-6 * max(spacing.max(), 1.0):
        print(
            "refine: warning - layer transform has shear "
            f"(off-diagonal {off_diagonal:.3g}); axis-aligned approximation used."
        )
    if np.any(spacing <= 0):
        raise ValueError("refine: non-positive voxel spacing derived from transform")
    return u, spacing

def _global_box_corners(center_global_voxel, half_extent_global_voxel):
    """The 8 corners of an axis-aligned global-voxel box."""
    offsets = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    return center_global_voxel + offsets * half_extent_global_voxel

def layer_shader_window(layer):
    """The layer's `normalized` shader range [lo, hi], or None if unset.

    This is the intensity window the user tuned in the viewer (the 'normalized'
    invlerp control). Reusing it for registration normalizes both layers on the
    same meaningful contrast the user sees, instead of each cutout's raw min/max.
    """
    controls = getattr(layer, "shader_controls", None)
    if not controls:
        return None
    normalized = controls.get("normalized")
    window = getattr(normalized, "range", None)
    if window is None:
        return None
    return float(window[0]), float(window[1])

def fetch_layer_image(layer, scale_global, global_scale_phys, center_global_voxel,
                      half_extent_global_voxel, mip=0, intensity_window=None):
    """Fetch a subvolume and return it as a SimpleITK image in global physical space.

    The returned image's origin / spacing / direction place every voxel at its
    true global physical position, so two layers fetched this way share the
    same physical frame and start roughly aligned.

    ``scale_global`` is the (normalized) global voxel size used for the
    SimpleITK physical frame; ``global_scale_phys`` is the global voxel size in
    metres, used to relate the source's native resolution to the global frame.
    """
    ndim = len(scale_global)
    matrix = layer_transform_matrix(layer, ndim)
    m_linear = matrix[:ndim, :ndim]
    m_translation = matrix[:ndim, ndim]

    dataset = _open_source(layer.source[0].url, mip)  # lazy: no voxel data read yet

    # The spatial axes are in the dataset's stored order, which matches the
    # column order of the layer transform; channel/time axes are dropped.
    domain = dataset.domain
    spatial, nonspatial = _spatial_axis_order(list(domain.labels), dataset.rank, ndim)

    # Array index -> intrinsic coordinate. The layer transform maps the source's
    # intrinsic (physical, in global-voxel units) coordinate to global voxels,
    # not the raw array index; this captures sources stored at a resolution
    # different from the global frame (e.g. a low-res OME-Zarr overview level).
    ratio, offset = _native_voxel_geometry(
        layer.source[0].url, mip, spatial, global_scale_phys)

    # Full array-index -> global-voxel affine: global = m @ (ratio * index + offset).
    linear = m_linear @ np.diag(ratio)
    translation = m_linear @ offset + m_translation
    affine_inv = np.linalg.inv(np.block(
        [[linear, translation[:, None]], [np.zeros((1, ndim)), np.ones((1, 1))]]))

    # Map the global-voxel box into array-index coordinates and take the
    # bounding integer range.
    corners_global = _global_box_corners(center_global_voxel, half_extent_global_voxel)
    corners_h = np.hstack([corners_global, np.ones((corners_global.shape[0], 1))])
    corners_local = (affine_inv @ corners_h.T).T[:, :ndim]
    local_min = np.floor(corners_local.min(axis=0)).astype(int)
    local_max = np.ceil(corners_local.max(axis=0)).astype(int) + 1

    # Clamp the requested array-index box to the array bounds on each spatial axis.
    lo = np.maximum(local_min, [int(domain[a].inclusive_min) for a in spatial])
    hi = np.minimum(local_max, [int(domain[a].exclusive_max) for a in spatial])
    if np.any(hi <= lo):
        raise RuntimeError(
            "refine: requested box does not overlap layer bounds "
            f"({layer.source[0].url}); check the landmark and box size."
        )

    # Read ONLY the small cutout into memory (the whole volume stays lazy).
    index = [slice(None)] * dataset.rank
    for axis in nonspatial:
        index[axis] = int(domain[axis].inclusive_min)
    for k, axis in enumerate(spatial):
        index[axis] = slice(int(lo[k]), int(hi[k]))
    array_local = np.asarray(dataset[tuple(index)])  # spatial axes, native order

    # numpy [a0, a1, a2] -> SimpleITK array order [a2, a1, a0]; SimpleITK image
    # index i then corresponds to local spatial axis i (matching the transform).
    array_rev = np.ascontiguousarray(np.transpose(array_local, tuple(range(ndim - 1, -1, -1))))
    image = sitk.GetImageFromArray(array_rev.astype(np.float32))

    # Geometry: linear_phys maps local voxel -> global physical (= global voxel * scale).
    diag_scale = np.diag(scale_global)
    linear_phys = diag_scale @ linear
    direction, spacing = polar_decompose(linear_phys)
    origin = linear_phys @ lo + diag_scale @ translation

    image.SetSpacing([float(s) for s in spacing])
    image.SetDirection([float(v) for v in direction.flatten()])
    image.SetOrigin([float(o) for o in origin])

    image = sitk.Cast(image, sitk.sitkFloat32)
    if intensity_window is not None:
        # Clip to the user's shader window, then map it to [0, 1]. This fixes the
        # normalization to the same contrast both layers are displayed with,
        # rather than each cutout's own (outlier-sensitive) min/max.
        lo, hi = intensity_window
        image = sitk.IntensityWindowing(
            image, windowMinimum=float(lo), windowMaximum=float(hi),
            outputMinimum=0.0, outputMaximum=1.0)
    else:
        image = sitk.RescaleIntensity(image, 0.0, 1.0)
    return image

def _set_metric(registration, metric):
    """Configure the similarity metric on a registration method.

    'correlation' (default) and 'meansquares' suit same-modality data (the
    common case here: two scans of the same specimen at different resolutions)
    and give a far stronger gradient on a small, blurry overview box than mutual
    information, which is meant for cross-modal registration and is near-flat
    here. 'mattes'/'mi' remains available for genuinely multi-modal pairs.
    """
    name = str(metric).lower()
    if name in ("correlation", "cc", "ncc"):
        registration.SetMetricAsCorrelation()
    elif name in ("meansquares", "ms", "msq"):
        registration.SetMetricAsMeanSquares()
    elif name in ("mattes", "mi", "mutualinformation"):
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    else:
        raise ValueError(
            f"refine: unknown metric '{metric}' "
            "(use 'correlation', 'meansquares', or 'mattes')")

def _registration_method(transform, metric="correlation"):
    """A SimpleITK registration configured for local refinement.

    The chosen metric (see `_set_metric`), full-box sampling, and a
    RegularStepGradientDescent optimizer with physical-shift scaling -- this
    combination is stable from a near-aligned start. The pyramid is kept gentle
    ([2, 1]); an aggressive coarse level blurs small subvolumes enough to
    diverge.
    """
    registration = sitk.ImageRegistrationMethod()
    _set_metric(registration, metric)
    registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    # Sigmas in voxel units: the shared physical frame is normalized so the
    # finest spacing is ~1, and voxel-unit sigmas avoid ITK's "suspiciously
    # small spacing" guard that trips on raw SI-metre scales.
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    registration.SetInitialTransform(transform, inPlace=True)
    return registration

def _model_transform(model, ndim, center):
    """Build the second-stage transform for the requested DOF model.

    'rigid' (default, 6 DOF) and 'similarity' (7 DOF, adds one isotropic scale)
    cannot represent shear, so they are the right constraint for two scans of
    the same specimen: an unconstrained 12-DOF 'affine' has nine linear DOF that
    a small, low-contrast box does not pin down, and the optimizer spends them on
    spurious shear/anisotropic scale instead of the true rigid motion. 'affine'
    remains available for genuinely non-rigid cases. Euler3D/Similarity3D are
    3D-only; 'affine' is the fallback for other dimensions.
    """
    name = str(model).lower()
    if name == "rigid" and ndim == 3:
        transform = sitk.Euler3DTransform()
    elif name == "similarity" and ndim == 3:
        transform = sitk.Similarity3DTransform()
    elif name == "affine":
        transform = sitk.AffineTransform(ndim)
    elif name in ("rigid", "similarity"):
        raise ValueError(f"refine: model '{model}' requires 3D data (got {ndim}D)")
    else:
        raise ValueError(
            f"refine: unknown model '{model}' "
            "(use 'rigid', 'similarity', or 'affine')")
    transform.SetCenter(center)
    return transform

def _fft_best_shift(fixed_array, moving_array, moving_mask, overlap_frac=0.3):
    """Exhaustive best integer translation by masked normalized cross-correlation.

    Returns ``(shift, ncc)``: the [a0, a1, a2] shift (in the arrays' own axis
    order) that best aligns `moving_array` to `fixed_array`, and the overlap
    normalized cross-correlation there (a ready alignment score, so the caller
    need not re-resample to grade the fit). FFT-based, so -- unlike a gradient
    optimizer -- it searches every translation at once and finds a shift of a
    hundred voxels as easily as one. `moving_mask` marks where the (resampled)
    moving has real data; the masked NCC (Padfield) uses it so the zero-fill left
    by resampling does not bias the correlation toward the coverage boundary (a
    plain cross-correlation locks onto that instead of the anatomy). The fixed
    array is treated as fully valid. Shifts whose overlap is below `overlap_frac`
    of the maximum are ignored so a sliver of overlap cannot win spuriously.
    """
    fixed = fixed_array.astype(np.float64)
    valid_fixed = np.ones_like(fixed)
    mask = moving_mask.astype(np.float64)
    moving = moving_array.astype(np.float64) * mask

    def correlate(a, b):
        return fftconvolve(a, b[::-1, ::-1, ::-1], mode="full")

    overlap = correlate(valid_fixed, mask)
    sum_fixed = correlate(fixed, mask)
    sum_moving = correlate(valid_fixed, moving)
    sum_fixed_sq = correlate(fixed * fixed, mask)
    sum_moving_sq = correlate(valid_fixed, moving * moving)
    sum_product = correlate(fixed, moving)

    overlap = np.maximum(overlap, 1e-9)
    numerator = sum_product - sum_fixed * sum_moving / overlap
    var_fixed = np.maximum(sum_fixed_sq - sum_fixed ** 2 / overlap, 0.0)
    var_moving = np.maximum(sum_moving_sq - sum_moving ** 2 / overlap, 0.0)
    denominator = np.sqrt(var_fixed * var_moving)
    ncc = np.where(denominator > 1e-9, numerator / np.maximum(denominator, 1e-9), 0.0)
    ncc[overlap < overlap_frac * overlap.max()] = 0.0

    peak = np.unravel_index(np.argmax(ncc), ncc.shape)
    shift = np.array(peak) - (np.array(moving_array.shape) - 1)
    return shift, float(ncc[peak])            # (shift [a0,a1,a2], overlap NCC at it)

def global_prealign(fixed_image, moving_image, max_angle_deg=8.0, angle_step_deg=2.0,
                    passes=1):
    """Coarse global rigid pre-alignment over 3D rotation and full translation.

    The local optimizer only converges from a near-aligned start, so a manual
    alignment that is off by a large rotation or translation (well beyond its
    capture range) leaves it wandering on noise. This searches rotations about all
    three axes in [-max_angle_deg, +max_angle_deg] (no assumed axis), pairing each
    with the globally optimal translation from `_fft_best_shift`, and returns the
    Euler3DTransform (fixed->moving) with the best honest overlap correlation to
    seed the local refinement. The three axes are swept by coordinate descent
    (each axis in turn, holding the others at their running best, repeated
    `passes` times) so the cost stays ~`passes * 3 * n_angles` rather than the
    cube of a full 3D grid; the local optimizer then polishes the coupling.

    The current alignment (identity) is the baseline candidate, and a rotated
    candidate is only returned if it strictly beats it: prealign is thus monotonic
    -- it can only improve on the starting alignment, never seed the local
    optimizer with something worse. Returns ``(None, baseline_score)`` when nothing
    beats the baseline, so `register_pair` falls back to its robust
    translation-first staging.

    Returns (transform_or_None, score).
    """
    center = [float(c) for c in fixed_image.TransformContinuousIndexToPhysicalPoint(
        [(sz - 1) / 2.0 for sz in fixed_image.GetSize()])]
    fixed_array = sitk.GetArrayViewFromImage(fixed_image).astype(np.float64)
    direction = np.array(fixed_image.GetDirection()).reshape(3, 3)
    spacing = np.array(fixed_image.GetSpacing())
    ones = sitk.Add(sitk.Cast(moving_image * 0, sitk.sitkFloat32), 1.0)

    def evaluate(angles_xyz):
        """Best transform (rotation angles_xyz + FFT translation) and its score.

        The masked-NCC peak from `_fft_best_shift` is the overlap correlation at
        the chosen shift, so it doubles as the score -- no extra resample needed.
        """
        radians = [np.deg2rad(a) for a in angles_xyz]
        rotation = sitk.Euler3DTransform()
        rotation.SetCenter(center)
        rotation.SetRotation(*radians)
        moving_rotated = sitk.GetArrayViewFromImage(sitk.Resample(
            moving_image, fixed_image, rotation, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)).astype(np.float64)
        coverage = sitk.GetArrayViewFromImage(sitk.Resample(
            ones, fixed_image, rotation, sitk.sitkNearestNeighbor, 0.0,
            sitk.sitkFloat32)) > 0.5
        shift, score = _fft_best_shift(fixed_array, moving_rotated, coverage)  # [a0,a1,a2]
        # array axes are [z, y, x]; physical shift folds in the fixed geometry.
        shift_phys = direction @ (spacing * shift[::-1])
        linear = np.array(rotation.GetMatrix()).reshape(3, 3)
        candidate = sitk.Euler3DTransform()
        candidate.SetCenter(center)
        candidate.SetRotation(*radians)
        candidate.SetTranslation([float(v) for v in -(linear @ shift_phys)])
        return score, candidate

    identity = sitk.Euler3DTransform()
    identity.SetCenter(center)
    best = (overlap_correlation(fixed_image, moving_image, sitk.Transform(identity)), None)

    steps = int(round(max_angle_deg / angle_step_deg))
    grid = [s * angle_step_deg for s in range(-steps, steps + 1)]
    current = [0.0, 0.0, 0.0]                          # working angles (x, y, z)
    for _ in range(passes):
        for axis in range(3):
            axis_best = (best[0], current[axis])
            for angle in grid:
                trial = list(current)
                trial[axis] = angle
                score, candidate = evaluate(trial)
                if score > best[0]:
                    best = (score, candidate)
                if score > axis_best[0]:
                    axis_best = (score, angle)
            current[axis] = axis_best[1]               # descend along this axis
    return best[1], best[0]

def register_pair(fixed_image, moving_image, metric="correlation", model="rigid",
                  initial_transform=None):
    """Register `moving` to `fixed` in their shared physical space.

    Returns the SimpleITK transform T mapping fixed physical points to moving
    physical points. The images already share a physical frame (both are
    fetched into global physical space), so the transform starts at identity.

    Registration is staged: a translation pass first (or, when `initial_transform`
    is given, that coarse pre-alignment), then the chosen `model` initialized from
    it. A from-scratch 12-DOF affine tends to misuse its extra degrees of freedom
    (spurious rotation/shear) on what is mostly a residual shift; staging, plus
    constraining `model` to 'rigid'/'similarity', converges far more reliably. The
    model is centered on the data: the cutout sits at large physical coordinates
    (its array-index origin is thousands of voxels from 0), so a transform centered
    at (0,0,0) would map tiny linear-parameter changes to huge point shifts, and
    SetOptimizerScalesFromPhysicalShift would then freeze the linear DOF so only
    the translation moves.

    `initial_transform` (e.g. from `global_prealign`) seeds the model with a
    rotation+translation the local optimizer could not reach on its own; when None
    the translation pass supplies the starting offset.
    """
    ndim = fixed_image.GetDimension()

    center = fixed_image.TransformContinuousIndexToPhysicalPoint(
        [(sz - 1) / 2.0 for sz in fixed_image.GetSize()])
    transform = _model_transform(model, ndim, center)
    if initial_transform is None:
        translation = sitk.TranslationTransform(ndim)
        _registration_method(translation, metric).Execute(fixed_image, moving_image)
        transform.SetTranslation(translation.GetOffset())
    else:
        # Seed the model to match the pre-alignment. Both share `center`, so
        # copying its orthonormal matrix and translation reproduces its action.
        transform.SetMatrix(initial_transform.GetMatrix())
        transform.SetTranslation(initial_transform.GetTranslation())
    _registration_method(transform, metric).Execute(fixed_image, moving_image)

    return sitk.Transform(transform)

def overlap_correlation(fixed_image, moving_image, transform):
    """Normalized cross-correlation of the two images on their overlap.

    Resamples `moving` onto the fixed grid through `transform` (fixed->moving
    physical) and returns the Pearson correlation over the region the moving
    image actually covers. Unlike the optimized metric -- which the optimizer can
    nudge down by fitting noise even when nothing lines up -- this is a bounded,
    interpretable [-1, 1] measure of real shared structure, so the guard uses it
    to tell a genuine match from wandering on a featureless box.
    """
    resampled = sitk.Resample(
        moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    ones = sitk.GetImageFromArray(
        np.ones(sitk.GetArrayViewFromImage(moving_image).shape, dtype=np.float32))
    ones.CopyInformation(moving_image)
    covered = sitk.Resample(
        ones, fixed_image, transform, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)

    fixed = sitk.GetArrayViewFromImage(fixed_image).ravel()
    moving = sitk.GetArrayViewFromImage(resampled).ravel()
    mask = sitk.GetArrayViewFromImage(covered).ravel() > 0.5
    if mask.sum() < 2:
        return 0.0
    a, b = fixed[mask], moving[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def transform_to_matrix(transform, ndim):
    """Convert a SimpleITK affine transform to a 4x4 numpy matrix.

    Recovered by sampling the transform's action on the origin and the unit
    basis points, so it works for any concrete affine transform type returned
    by ImageRegistrationMethod.Execute (which is a generic Transform).
    """
    origin = np.array(transform.TransformPoint([0.0] * ndim))
    matrix = np.eye(ndim + 1)
    for axis in range(ndim):
        basis = [0.0] * ndim
        basis[axis] = 1.0
        matrix[:ndim, axis] = np.array(transform.TransformPoint(basis)) - origin
    matrix[:ndim, ndim] = origin
    return matrix

def _is_image_layer(layer):
    return (isinstance(layer.layer, neuroglancer.ImageLayer)
            or getattr(layer.layer, "type", None) == "image")

def resolve_roles(state, fixed=None, moving=None):
    """Determine the (fixed_name, moving_name) layer pair.

    Priority: explicit arguments, then ref:: / mov:: name prefixes, then the
    two-image-layer fallback (one image layer is the reference, the selected
    layer is moving).
    """
    names = [l.name for l in state.layers]

    if fixed is not None and moving is not None:
        return fixed, moving

    tagged_ref = [n for n in names if n.startswith(REFERENCE_PREFIX)]
    tagged_mov = [n for n in names if n.startswith(MOVING_PREFIX)]
    if len(tagged_ref) == 1 and len(tagged_mov) == 1:
        return tagged_ref[0], tagged_mov[0]
    if tagged_ref or tagged_mov:
        raise ValueError(
            "refine: expected exactly one 'ref::' and one 'mov::' tagged layer, "
            f"found ref={tagged_ref}, mov={tagged_mov}. "
            "Tag layers with the alt+r / alt+m keybindings."
        )

    image_layers = [l.name for l in state.layers if _is_image_layer(l)]
    if moving is None:
        moving = state.selectedLayer.layer
    others = [n for n in image_layers if n != moving]
    if fixed is None:
        if len(others) == 1:
            fixed = others[0]
        else:
            raise ValueError(
                "refine: cannot infer the reference layer (image layers: "
                f"{image_layers}). Tag layers with alt+r / alt+m, or pass "
                "fixed= and moving= explicitly."
            )
    return fixed, moving

def landmark_point(state, ndim):
    """Return the box-center point in global voxel coordinates.

    Uses the single __LANDMARK__ annotation if present (see
    place_translation_landmark), otherwise the viewer center voxel_coordinates.
    """
    if "__LANDMARK__" in state.layers:
        annotations = state.layers["__LANDMARK__"].annotations
        if len(annotations) >= 1:
            return np.array(annotations[0].point, dtype=float)
    return np.array(state.voxel_coordinates, dtype=float)

def refine_registration(size_voxels=200, fixed=None, moving=None, mip=0, apply=True,
                        metric="correlation", model="rigid",
                        fixed_mip=None, moving_mip=None, use_shader_window=True,
                        guard=True, max_shift_voxels=None, min_correlation=0.1,
                        prealign=True, prealign_max_angle=8.0, prealign_step=2.0):
    """Refine the moving layer's registration around the landmark.

    Callable from the interactive (`python -i`) session. Fetches a cube of
    `size_voxels` global voxels (scalar, or per-axis in viewer.dimensions order)
    centered on the landmark from both the reference and moving layers, runs a
    SimpleITK registration, and (if `apply`) composes the correction onto the
    moving layer's transform via apply_transform_to_layer.

    `metric` selects the similarity measure ('correlation', 'meansquares', or
    'mattes'); `model` constrains the degrees of freedom ('rigid', 'similarity',
    or 'affine'). The defaults (correlation + rigid) suit same-specimen scans and
    avoid the spurious shear an unconstrained affine produces on a small box.

    `mip` picks the multiscale level (0 = full resolution) fetched from each
    source; `fixed_mip` / `moving_mip` override it per layer, so a coarse overview
    and a fine VOI can be compared at whichever levels give a comparable working
    resolution. The physical framing (`_native_voxel_geometry`) adapts to each
    chosen level, so the two cutouts still share a frame regardless of the levels.

    `use_shader_window` normalizes each cutout by the layer's viewer shader window
    (the 'normalized' invlerp range) instead of its raw min/max, so both layers
    use the same meaningful contrast; falls back to min/max when a layer has no
    shader window. Set False to force min/max normalization.

    `prealign` (default on) runs `global_prealign` first: a coarse exhaustive
    search over z-rotation (+-`prealign_max_angle` degrees, step `prealign_step`)
    and full FFT translation, seeding the local optimizer with a rotation and
    translation it could not otherwise reach. This is what recovers a manual
    alignment that is off by several degrees and many voxels (the local optimizer
    only has a small capture range and would just wander). Set False to start the
    local optimizer from the current alignment.

    `guard` (default on) rejects a result whose overlap cross-correlation stays
    below `min_correlation` (default 0.1 -- essentially no shared structure) or
    that moves the box by more than `max_shift_voxels` (default half the smallest
    box side). On a featureless landmark box the optimized metric can
    still be nudged down by fitting noise, so the guard judges real alignment with
    an independent cross-correlation and applies nothing rather than wandering off
    the (already-good) manual alignment. Set False to always apply the raw result.
    With `prealign` the box displacement can legitimately be large, so raise
    `max_shift_voxels` when recovering a big manual error.

    Returns the 4x4 global-voxel correction matrix that was applied (identity if
    the guard rejected the result).
    """
    global viewer
    state = viewer.state
    dims = state.dimensions
    # Per-axis physical size of one global voxel (metres), in the dimensions'
    # order. Used both to relate each source's native resolution to the global
    # frame (in fetch_layer_image) and, after normalizing to the finest axis,
    # as the SimpleITK physical frame. Normalizing keeps the anisotropy ratio
    # but gives SimpleITK sane O(1) spacings instead of the ~1e-9 values that
    # trip ITK's small-spacing guard (any common factor cancels in the
    # round-trip below).
    global_scale_phys = np.array(dims.scales, dtype=float) * np.array(
        [_unit_to_meters(u) for u in dims.units], dtype=float)
    if global_scale_phys.size == 0:
        raise ValueError(
            "refine: the viewer has no global dimensions (viewer.state.dimensions "
            "is empty), so there is no coordinate frame to register in. Load a "
            "state that has top-level 'dimensions' (e.g. python -i ngregister.py "
            "--url ...), or set viewer.dimensions, then retry.")
    scale_global = global_scale_phys / np.min(global_scale_phys)
    ndim = len(global_scale_phys)

    fixed_name, moving_name = resolve_roles(state, fixed=fixed, moving=moving)
    fixed_mip = mip if fixed_mip is None else int(fixed_mip)
    moving_mip = mip if moving_mip is None else int(moving_mip)
    print(f"refine: reference='{fixed_name}' (mip {fixed_mip})  "
          f"moving='{moving_name}' (mip {moving_mip})")

    center_global_voxel = landmark_point(state, ndim)

    size_voxels = np.broadcast_to(np.asarray(size_voxels, dtype=float), (ndim,))
    half_extent_global_voxel = size_voxels / 2.0

    fixed_layer = state.layers[fixed_name].layer
    moving_layer = state.layers[moving_name].layer
    fixed_window = layer_shader_window(fixed_layer) if use_shader_window else None
    moving_window = layer_shader_window(moving_layer) if use_shader_window else None

    fixed_image = fetch_layer_image(
        fixed_layer, scale_global, global_scale_phys,
        center_global_voxel, half_extent_global_voxel, mip=fixed_mip,
        intensity_window=fixed_window,
    )
    moving_image = fetch_layer_image(
        moving_layer, scale_global, global_scale_phys,
        center_global_voxel, half_extent_global_voxel, mip=moving_mip,
        intensity_window=moving_window,
    )

    initial_transform = None
    if prealign:
        initial_transform, prealign_score = global_prealign(
            fixed_image, moving_image,
            max_angle_deg=prealign_max_angle, angle_step_deg=prealign_step)
        if initial_transform is None:
            print("refine: prealign found no rotation better than the current "
                  f"alignment (overlap correlation {prealign_score:.3g}); "
                  "starting the local optimizer from it")
        else:
            angles = np.rad2deg([initial_transform.GetAngleX(),
                                 initial_transform.GetAngleY(),
                                 initial_transform.GetAngleZ()])
            print(f"refine: prealign rotation (x,y,z) "
                  f"{angles[0]:+.1f},{angles[1]:+.1f},{angles[2]:+.1f} deg "
                  f"(overlap correlation {prealign_score:.3g})")

    transform = register_pair(fixed_image, moving_image, metric=metric, model=model,
                              initial_transform=initial_transform)

    # T maps fixed physical -> moving physical. The spatial correction to apply
    # to the moving layer is C = T^-1 (move the moving feature now at T(p) to p).
    t_phys = transform_to_matrix(transform, ndim)
    c_phys = np.linalg.inv(t_phys)

    # Global physical -> global voxel: phys = diag(scale) @ voxel.
    diag_scale = np.diag(scale_global)
    diag_scale_inv = np.diag(1.0 / scale_global)
    c_vox = np.eye(ndim + 1)
    c_vox[:ndim, :ndim] = diag_scale_inv @ c_phys[:ndim, :ndim] @ diag_scale
    c_vox[:ndim, ndim] = diag_scale_inv @ c_phys[:ndim, ndim]

    # Report the actual displacement the correction induces over the box, not
    # the homogeneous translation column: for a rotation about a point far from
    # the origin that column is large even when the box barely moves, which is
    # misleading. Sample the landmark and the box corners.
    probes = np.vstack([
        _global_box_corners(center_global_voxel, half_extent_global_voxel),
        center_global_voxel])
    probes_h = np.hstack([probes, np.ones((probes.shape[0], 1))])
    moved = (c_vox @ probes_h.T).T[:, :ndim]
    disp = np.linalg.norm(moved - probes, axis=1)

    if guard:
        # Judge real alignment with an independent cross-correlation: the
        # optimized metric can be nudged down by fitting noise on a featureless
        # box, but genuine shared structure shows up as overlap correlation.
        correlation = overlap_correlation(fixed_image, moving_image, transform)
        # Allow up to half the box (beyond that the overlap vanishes); prealign
        # legitimately recovers large manual errors, and the correlation gate is
        # the real validator of whether a big correction is trustworthy.
        limit = (float(np.min(size_voxels)) / 2.0 if max_shift_voxels is None
                 else float(max_shift_voxels))
        reason = None
        if not np.isfinite(correlation) or abs(correlation) < min_correlation:
            reason = (f"overlap correlation {correlation:.3g} < {min_correlation:g}; "
                      "the box lacks structure shared by both layers")
        elif disp.max() > limit:
            reason = (f"correction moves the box {disp.max():.3g} voxels, over the "
                      f"{limit:.3g}-voxel guard limit")
        if reason is not None:
            print(f"refine: rejected correction -- {reason}. "
                  "Nothing applied (pass guard=False, lower min_correlation, or "
                  "raise max_shift_voxels to override).")
            return np.eye(ndim + 1)
        print(f"refine: overlap correlation {correlation:.3g}")

    print(f"refine: applying correction (landmark moves {disp[-1]:.3g} voxels, "
          f"max over box {disp.max():.3g} voxels)")

    if apply:
        apply_transform_to_layer(c_vox, chain_backwards=False, layer_name=moving_name)
        save_state_in_history()

    return c_vox

def _resolve_target_layer(state, target):
    """Name of the layer whose space to chain into.

    Explicit `target`, else the single ref:: tagged layer, else the selected
    layer. Must be an image layer (it needs a source transform).
    """
    if target is None:
        tagged = [l.name for l in state.layers if l.name.startswith(REFERENCE_PREFIX)]
        target = tagged[0] if len(tagged) == 1 else state.selectedLayer.layer
    image_layers = [l.name for l in state.layers if _is_image_layer(l)]
    if target not in state.layers:
        raise ValueError(
            f"chain: target '{target}' is not a layer. Image layers: {image_layers}. "
            "Pass target='<name>'.")
    if target not in image_layers:
        detected = getattr(state.layers[target].layer, "type", None)
        raise ValueError(
            f"chain: target '{target}' is not an image layer (type={detected!r}). "
            f"Image layers: {image_layers}. Pass target='<name>'.")
    return target

def chain_to_layer_space(target=None, apply=True):
    """Re-express every image layer in one chosen image's coordinate space.

    Callable from the interactive (`python -i`) session, like refine_registration.
    Picks a target image layer (`target` name, else the single ref:: tagged
    layer, else the selected layer) and left-multiplies every image layer's
    source[0].transform by inv(M_target). The target layer's matrix therefore
    becomes the identity and every other layer is expressed relative to it; the
    global dimensions (scales/units) are unchanged. The single __LANDMARK__ point
    annotation is mapped by the same inv(M_target) so it stays on its feature.

    Returns the 4x4 inv(M_target) that was applied.
    """
    global viewer
    state = viewer.state
    ndim = len(state.dimensions.scales)
    if ndim == 0:
        raise ValueError(
            "chain: the viewer has no global dimensions (viewer.state.dimensions "
            "is empty); refusing to rewrite layer transforms, which would corrupt "
            "the state. Load a state with top-level 'dimensions' first.")
    target_name = _resolve_target_layer(state, target)
    target_inverse = np.linalg.inv(
        layer_transform_matrix(state.layers[target_name].layer, ndim))

    if not apply:
        print(f"chain: target '{target_name}' (dry run, nothing written).")
        return target_inverse

    save_state_in_history()                            # restore point before the rewrite
    with viewer.txn() as v:
        output_dimensions = v.dimensions.to_json()
        # Collect names first, then index by name to rewrite each source -- the
        # same pattern apply_transform_to_layer uses. Reassigning `.source` while
        # iterating `v.layers` can corrupt the layer list when the change syncs to
        # the browser.
        image_names = [l.name for l in v.layers if _is_image_layer(l)]
        for name in image_names:
            layer = v.layers[name].layer
            rebased = target_inverse @ layer_transform_matrix(layer, ndim)
            transform = neuroglancer.CoordinateSpaceTransform(
                {"matrix": rebased[:ndim, : ndim + 1].tolist(),
                 "outputDimensions": output_dimensions})
            v.layers[name].layer.source[0] = neuroglancer.LayerDataSource(
                {"url": layer.source[0].url, "transform": transform.to_json()})

        if "__LANDMARK__" in v.layers and len(v.layers["__LANDMARK__"].annotations) >= 1:
            annotation = v.layers["__LANDMARK__"].annotations[0]
            point = np.array(annotation.point, dtype=float)
            moved = (target_inverse @ np.append(point, 1.0))[:ndim]
            v.layers["__LANDMARK__"].annotations[0] = neuroglancer.PointAnnotation(
                id=annotation.id, point=moved.tolist())

    print(f"chain: '{target_name}' is now the identity space; "
          "all image layers and the landmark re-expressed relative to it.")
    return target_inverse

def print_last_state():
    print(neuroglancer.to_url(viewer.state))

atexit.register(print_last_state)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_state_arguments(ap, required=False)
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    if args.state:
        viewer.set_state(args.state)

    print(viewer)

    webbrowser.open_new(viewer.get_viewer_url())

    # Binding the actions
    viewer.actions.add('rotate-layer-absolute-z-cw', rotate_layer_absolute_z_clockwise)
    viewer.actions.add('rotate-layer-absolute-y-cw', rotate_layer_absolute_y_clockwise)
    viewer.actions.add('rotate-layer-absolute-x-cw', rotate_layer_absolute_x_clockwise)

    viewer.actions.add('rotate-layer-absolute-z-ccw', rotate_layer_absolute_z_counterclockwise)
    viewer.actions.add('rotate-layer-absolute-y-ccw', rotate_layer_absolute_y_counterclockwise)
    viewer.actions.add('rotate-layer-absolute-x-ccw', rotate_layer_absolute_x_counterclockwise)

    viewer.actions.add('translate-layer-with-cursor', translate_layer_from_cursor_to_center)
    viewer.actions.add('place-translation-landmark', place_translation_landmark)
    viewer.actions.add('translate-layer-with-landmark', translate_layer_to_landmark)
    viewer.actions.add('flip-x-axis', flip_x_axis)
    viewer.actions.add('flip-y-axis', flip_y_axis)
    viewer.actions.add('flip-z-axis', flip_z_axis)
    viewer.actions.add('mark-layer-moving', mark_layer_moving)
    viewer.actions.add('mark-layer-reference', mark_layer_reference)
    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer['keyt'] = 'translate-layer-with-cursor'
        s.input_event_bindings.viewer['keyy'] = 'place-translation-landmark'
        s.input_event_bindings.viewer['shift+keyt'] = 'translate-layer-with-landmark'
        s.input_event_bindings.viewer['keyi'] = 'rotate-layer-absolute-z-cw'
        s.input_event_bindings.viewer['keyj'] = 'rotate-layer-absolute-x-cw'
        s.input_event_bindings.viewer['keyk'] = 'rotate-layer-absolute-y-cw'
        s.input_event_bindings.viewer['shift+keyi'] = 'rotate-layer-absolute-z-ccw'
        s.input_event_bindings.viewer['shift+keyj'] = 'rotate-layer-absolute-x-ccw'
        s.input_event_bindings.viewer['shift+keyk'] = 'rotate-layer-absolute-y-ccw'
        s.input_event_bindings.viewer['alt+keyx'] = 'flip-x-axis'
        s.input_event_bindings.viewer['alt+keyy'] = 'flip-y-axis'
        s.input_event_bindings.viewer['alt+keyz'] = 'flip-z-axis'
        s.input_event_bindings.viewer['alt+keym'] = 'mark-layer-moving'
        s.input_event_bindings.viewer['alt+keyr'] = 'mark-layer-reference'