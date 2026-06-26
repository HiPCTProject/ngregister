import argparse
import webbrowser

import neuroglancer
import neuroglancer.cli
import numpy as np
from scipy.spatial.transform import Rotation
import datetime
import atexit

import SimpleITK as sitk
from cloudvolume import CloudVolume

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
# Registration refinement (CloudVolume + SimpleITK)
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
# - CloudVolume: vol[x0:x1, y0:y1, z0:z1] returns a numpy array indexed
#   [x, y, z, channel] on the layer's local voxel grid (x-first).
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

def mark_active_layer(prefix):
    global viewer
    with viewer.txn() as v:
        name = v.selectedLayer.layer
        if name is None or name not in v.layers:
            return
        v.layers[name].name = prefix + _strip_role_prefix(name)

def mark_layer_moving(s):
    mark_active_layer(MOVING_PREFIX)

def mark_layer_reference(s):
    mark_active_layer(REFERENCE_PREFIX)

def source_url_to_cloudpath(url):
    """Turn a Neuroglancer source URL into a CloudVolume cloudpath.

    Strips a trailing "|<datatype>:" annotation used by newer Neuroglancer
    zarr/n5 sources (e.g. "...data.ome.zarr|zarr:") and keeps the
    "protocol://..." cloudpath that CloudVolume understands.
    """
    return url.split("|", 1)[0]

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

def fetch_layer_image(layer, scale_global, center_global_voxel,
                      half_extent_global_voxel, mip=0):
    """Fetch a subvolume and return it as a SimpleITK image in global physical space.

    The returned image's origin / spacing / direction place every voxel at its
    true global physical position, so two layers fetched this way share the
    same physical frame and start roughly aligned.
    """
    ndim = len(scale_global)
    matrix = layer_transform_matrix(layer, ndim)
    linear = matrix[:ndim, :ndim]
    translation = matrix[:ndim, ndim]
    matrix_inv = np.linalg.inv(matrix)

    # Map the global-voxel box into local voxel coordinates and take the
    # bounding integer range.
    corners_global = _global_box_corners(center_global_voxel, half_extent_global_voxel)
    corners_h = np.hstack([corners_global, np.ones((corners_global.shape[0], 1))])
    corners_local = (matrix_inv @ corners_h.T).T[:, :ndim]
    local_min = np.floor(corners_local.min(axis=0)).astype(int)
    local_max = np.ceil(corners_local.max(axis=0)).astype(int) + 1

    cloudpath = source_url_to_cloudpath(layer.source[0].url)
    try:
        vol = CloudVolume(cloudpath, mip=mip, fill_missing=True,
                          progress=False, use_https=True)
    except Exception as exc:  # surface a clear, layer-scoped error
        raise RuntimeError(
            f"refine: CloudVolume could not open source '{cloudpath}': {exc}"
        ) from exc

    # CloudVolume bounds are in mip-0-equivalent absolute voxel coordinates,
    # which we treat as the layer's local voxel grid. Clamp the request.
    bounds = vol.bounds  # Bbox in this mip's voxels
    lo = np.maximum(local_min, np.array(bounds.minpt[:ndim]))
    hi = np.minimum(local_max, np.array(bounds.maxpt[:ndim]))
    if np.any(hi <= lo):
        raise RuntimeError(
            "refine: requested box does not overlap layer bounds "
            f"({cloudpath}); check the landmark and box size."
        )

    cutout = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    array_xyz = np.asarray(cutout)[..., 0]  # drop channel -> [x, y, z]

    # numpy [x, y, z] -> SimpleITK array order [z, y, x] -> Image indexed (x,y,z).
    array_zyx = np.ascontiguousarray(np.transpose(array_xyz, (2, 1, 0)))
    image = sitk.GetImageFromArray(array_zyx.astype(np.float32))

    # Geometry: linear_phys maps local voxel -> global physical (= global voxel * scale).
    diag_scale = np.diag(scale_global)
    linear_phys = diag_scale @ linear
    direction, spacing = polar_decompose(linear_phys)
    origin = linear_phys @ lo + diag_scale @ translation

    image.SetSpacing([float(s) for s in spacing])
    image.SetDirection([float(v) for v in direction.flatten()])
    image.SetOrigin([float(o) for o in origin])

    image = sitk.Cast(sitk.RescaleIntensity(image, 0.0, 1.0), sitk.sitkFloat32)
    return image

def _registration_method(transform):
    """A SimpleITK registration configured for local refinement.

    Mattes MI (works across modalities), full-box sampling, and a
    RegularStepGradientDescent optimizer with physical-shift scaling -- this
    combination is stable from a near-aligned start. The pyramid is kept gentle
    ([2, 1]); an aggressive coarse level blurs small subvolumes enough to
    diverge.
    """
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
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

def register_affine(fixed_image, moving_image):
    """Affine-register `moving` to `fixed` in their shared physical space.

    Returns the SimpleITK transform T mapping fixed physical points to moving
    physical points. The images already share a physical frame (both are
    fetched into global physical space), so the transform starts at identity.

    Registration is staged: a translation pass first, then a full affine
    initialized from it. A 12-DOF affine optimized from scratch tends to misuse
    its extra degrees of freedom (spurious rotation/shear) on what is mostly a
    residual shift; the staged form converges far more reliably.
    """
    ndim = fixed_image.GetDimension()

    translation = sitk.TranslationTransform(ndim)
    _registration_method(translation).Execute(fixed_image, moving_image)

    affine = sitk.AffineTransform(ndim)
    affine.SetTranslation(translation.GetOffset())
    _registration_method(affine).Execute(fixed_image, moving_image)

    return sitk.Transform(affine)

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
    return getattr(layer.layer, "type", None) == "image"

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

def refine_registration(size_voxels=200, fixed=None, moving=None, mip=0, apply=True):
    """Refine the moving layer's registration around the landmark.

    Callable from the interactive (`python -i`) session. Fetches a cube of
    `size_voxels` global voxels (scalar, or per-axis in viewer.dimensions order)
    centered on the landmark from both the reference and moving layers, runs an
    affine SimpleITK registration, and (if `apply`) composes the correction onto
    the moving layer's transform via apply_transform_to_layer.

    Returns the 4x4 global-voxel correction matrix that was applied.
    """
    global viewer
    state = viewer.state
    dims = state.dimensions
    # Per-axis physical size of one global voxel, in the dimensions' order.
    # dims.scales are SI metres; they are only ever used as a consistent
    # voxel<->physical conversion factor (any common factor cancels in the
    # round-trip below), so normalize to the finest axis. This keeps the
    # anisotropy ratio but gives SimpleITK sane O(1) spacings instead of the
    # ~1e-9 values that trip ITK's small-spacing guard.
    scale_global = np.array(dims.scales, dtype=float)
    scale_global = scale_global / np.min(scale_global)
    ndim = len(scale_global)

    fixed_name, moving_name = resolve_roles(state, fixed=fixed, moving=moving)
    print(f"refine: reference='{fixed_name}'  moving='{moving_name}'")

    center_global_voxel = landmark_point(state, ndim)

    size_voxels = np.broadcast_to(np.asarray(size_voxels, dtype=float), (ndim,))
    half_extent_global_voxel = size_voxels / 2.0

    fixed_image = fetch_layer_image(
        state.layers[fixed_name].layer, scale_global,
        center_global_voxel, half_extent_global_voxel, mip=mip,
    )
    moving_image = fetch_layer_image(
        state.layers[moving_name].layer, scale_global,
        center_global_voxel, half_extent_global_voxel, mip=mip,
    )

    transform = register_affine(fixed_image, moving_image)

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

    shift_voxels = np.linalg.norm(c_vox[:ndim, ndim])
    print(f"refine: applying correction (translation {shift_voxels:.3g} voxels)")

    if apply:
        apply_transform_to_layer(c_vox, chain_backwards=False, layer_name=moving_name)
        save_state_in_history()

    return c_vox

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