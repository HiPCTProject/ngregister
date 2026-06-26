"""Tests for the registration refinement in ngregister.py.

The end-to-end test builds two real precomputed volumes with
``CloudVolume.from_numpy`` on the in-memory ``mem://`` backend (no disk), points
a Neuroglancer state at them, and checks that ``refine_registration`` recovers a
known injected shift. Running through real CloudVolume exercises the actual
``[x, y, z, channel]`` axis ordering that the pipeline has to invert.

Run with: ``pytest test_ngregister.py``
"""

import types

import numpy as np
import scipy.ndimage as ndi
import neuroglancer
from cloudvolume import CloudVolume

import ngregister


def _texture(shape=(100, 100, 100), seed=7):
    """A smooth, bandlimited random volume -- good signal for the MI metric."""
    rng = np.random.default_rng(seed)
    vol = ndi.gaussian_filter(rng.random(shape).astype(np.float32), 3.5)
    vol -= vol.min()
    vol /= vol.max()
    return vol


def _make_mem_volume(path, array, resolution):
    """Materialize a uint16 precomputed image at a mem:// cloudpath."""
    data = np.ascontiguousarray((array * 60000).astype(np.uint16))
    CloudVolume.from_numpy(
        data, vol_path=path, resolution=resolution,
        chunk_size=(64, 64, 64), layer_type="image", progress=False,
    )


def _state_with_layers(fixed_path, moving_path, scales, landmark):
    state = neuroglancer.ViewerState()
    state.dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=scales,
    )
    state.position = landmark
    state.layers.append(
        name="ref::fix",
        layer=neuroglancer.ImageLayer(source="precomputed://" + fixed_path),
    )
    state.layers.append(
        name="mov::mov",
        layer=neuroglancer.ImageLayer(source="precomputed://" + moving_path),
    )
    state.layers.append(
        name="__LANDMARK__",
        layer=neuroglancer.LocalAnnotationLayer(dimensions=state.dimensions),
    )
    state.layers["__LANDMARK__"].annotations.append(
        neuroglancer.PointAnnotation(id="lm", point=landmark),
    )
    return state


def test_refine_registration_recovers_known_shift(monkeypatch):
    # Anisotropic voxel size (z twice x/y) to exercise the scale handling.
    scales = [4, 4, 8]
    shift_voxels = np.array([3.0, -2.0, 1.0])

    vol = _texture()
    moving = ndi.shift(vol, shift_voxels, order=1, mode="reflect")

    fixed_path = "mem://ngregister-test/fixed"
    moving_path = "mem://ngregister-test/moving"
    _make_mem_volume(fixed_path, vol, scales)
    _make_mem_volume(moving_path, moving, scales)

    state = _state_with_layers(fixed_path, moving_path, scales, landmark=[50, 50, 50])
    monkeypatch.setattr(ngregister, "viewer", types.SimpleNamespace(state=state))

    correction = ngregister.refine_registration(size_voxels=60, apply=False)

    # The moving content is displaced by +shift; the correction must cancel it.
    recovered = correction[:3, 3]
    assert np.linalg.norm(recovered - (-shift_voxels)) < 1.0
    # A pure translation: the affine's linear part stays ~identity.
    assert np.allclose(correction[:3, :3], np.eye(3), atol=0.04)


def test_source_url_to_cloudpath_strips_datatype_suffix():
    assert ngregister.source_url_to_cloudpath(
        "https://h/d.ome.zarr|zarr:") == "https://h/d.ome.zarr"
    assert ngregister.source_url_to_cloudpath(
        "precomputed://gs://b/p") == "precomputed://gs://b/p"


def test_resolve_roles_prefers_name_prefixes():
    state = neuroglancer.ViewerState()
    state.dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[4, 4, 4])
    for name in ["ref::a", "mov::b", "other"]:
        state.layers.append(
            name=name, layer=neuroglancer.ImageLayer(source="precomputed://x"))
    assert ngregister.resolve_roles(state) == ("ref::a", "mov::b")


def test_resolve_roles_two_layer_fallback():
    state = neuroglancer.ViewerState()
    state.dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[4, 4, 4])
    for name in ["a", "b"]:
        state.layers.append(
            name=name, layer=neuroglancer.ImageLayer(source="precomputed://x"))
    state.selectedLayer.layer = "b"
    assert ngregister.resolve_roles(state) == ("a", "b")
