"""Tests for the registration refinement in ngregister.py.

The end-to-end tests build real volumes with TensorStore (precomputed and
sharded zarr3) under a temporary directory, point a Neuroglancer state at them,
and check that ``refine_registration`` recovers a known injected shift. Running
through real TensorStore exercises the lazy open + small-cutout read path and
the axis handling the pipeline has to invert -- including a zarr3 volume stored
in reversed ``z, y, x`` order with a permuting layer transform.

Run with: ``pytest test_ngregister.py``
"""

import json
import types

import numpy as np
import scipy.ndimage as ndi
import neuroglancer
import tensorstore as ts

import ngregister


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _texture(shape=(100, 100, 100), seed=7):
    """A smooth, bandlimited random volume -- good signal for the MI metric."""
    rng = np.random.default_rng(seed)
    vol = ndi.gaussian_filter(rng.random(shape).astype(np.float32), 3.5)
    vol -= vol.min()
    vol /= vol.max()
    return vol


def _u16(array):
    return np.ascontiguousarray((array * 60000).astype(np.uint16))


def _write_precomputed(dirpath, array_xyz, resolution):
    """Write a uint16 precomputed image (stored [x, y, z]); return source URL."""
    data = _u16(array_xyz)
    dataset = ts.open({
        "driver": "neuroglancer_precomputed",
        "kvstore": "file://" + str(dirpath) + "/",
        "create": True, "delete_existing": True,
        "scale_metadata": {"size": list(data.shape), "encoding": "raw",
                           "resolution": list(resolution), "chunk_size": [64, 64, 64]},
        "multiscale_metadata": {"data_type": "uint16", "num_channels": 1,
                                "type": "image"},
    }).result()
    dataset[:, :, :, 0] = data
    return "precomputed://file://" + str(dirpath) + "/"


def _write_zarr3_sharded(dirpath, array_native, dimension_names):
    """Write a sharded zarr3 array (stored in dimension_names order); return URL."""
    data = _u16(array_native)
    shape = list(data.shape)
    inner = [min(50, s) for s in shape]              # multiple inner chunks per shard
    shard = [(s + i - 1) // i * i for s, i in zip(shape, inner)]
    ts.open({
        "driver": "zarr3",
        "kvstore": "file://" + str(dirpath) + "/",
        "create": True, "delete_existing": True,
        "metadata": {
            "shape": shape,
            "data_type": "uint16",
            "chunk_grid": {"name": "regular",
                           "configuration": {"chunk_shape": shard}},
            "codecs": [{"name": "sharding_indexed", "configuration": {
                "chunk_shape": inner,
                "codecs": [{"name": "bytes",
                            "configuration": {"endian": "little"}}]}}],
            "dimension_names": dimension_names,
        },
    }).result()[...] = data
    return "zarr3://file://" + str(dirpath) + "/"


def _write_ome_zarr3_group(dirpath, array_xyz, resolution):
    """Write a minimal OME-Zarr v3 multiscale group (level 0 array); return URL.

    The source URL points at the *group*, so opening it as an array fails and
    the refinement must resolve the level-0 dataset path from the metadata.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    _write_zarr3_sharded(dirpath / "0", array_xyz, ["x", "y", "z"])
    group_meta = {
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"multiscales": [{"datasets": [{"path": "0"}]}]}},
    }
    (dirpath / "zarr.json").write_text(json.dumps(group_meta))
    return "zarr3://file://" + str(dirpath) + "/"


def _write_ome_zarr3_scaled(dirpath, array_xyz, scale, translation, unit="nanometer"):
    """Write an OME-Zarr v3 multiscale group whose level-0 voxel size is ``scale``.

    Unlike ``_write_ome_zarr3_group`` this records real ``coordinateTransformations``
    so the source's native resolution differs from the global frame -- the
    situation that broke the box mapping in issue #5.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    _write_zarr3_sharded(dirpath / "0", array_xyz, ["x", "y", "z"])
    group_meta = {
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"multiscales": [{
            "axes": [{"name": n, "type": "space", "unit": unit}
                     for n in ["x", "y", "z"]],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": list(scale)},
                {"type": "translation", "translation": list(translation)}]}],
        }]}},
    }
    (dirpath / "zarr.json").write_text(json.dumps(group_meta))
    return "zarr3://file://" + str(dirpath) + "/"


def _state(scales, landmark, fixed_layer, moving_layer, units="nm"):
    state = neuroglancer.ViewerState()
    state.dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units=units, scales=scales)
    state.position = landmark
    state.layers.append(name="ref::fix", layer=fixed_layer)
    state.layers.append(name="mov::mov", layer=moving_layer)
    state.layers.append(
        name="__LANDMARK__",
        layer=neuroglancer.LocalAnnotationLayer(dimensions=state.dimensions))
    state.layers["__LANDMARK__"].annotations.append(
        neuroglancer.PointAnnotation(id="lm", point=landmark))
    return state


def _image_layer(url, matrix=None, dims=None):
    if matrix is None:
        return neuroglancer.ImageLayer(source=url)
    source = neuroglancer.LayerDataSource(
        url=url,
        transform=neuroglancer.CoordinateSpaceTransform(
            matrix=matrix, output_dimensions=dims))
    return neuroglancer.ImageLayer(source=source)


def _assert_cancels_shift(correction, shift_voxels):
    # The moving content is displaced by +shift; the correction must cancel it.
    assert np.linalg.norm(correction[:3, 3] - (-shift_voxels)) < 1.0
    assert np.allclose(correction[:3, :3], np.eye(3), atol=0.04)


# ---------------------------------------------------------------------------
# End-to-end tests (real TensorStore volumes)
# ---------------------------------------------------------------------------

def test_refine_precomputed_recovers_shift(monkeypatch, tmp_path):
    scales = [4, 4, 8]                                 # anisotropic z
    shift = np.array([3.0, -2.0, 1.0])
    vol = _texture()
    moving = ndi.shift(vol, shift, order=1, mode="reflect")

    fixed_url = _write_precomputed(tmp_path / "fixed", vol, scales)
    moving_url = _write_precomputed(tmp_path / "moving", moving, scales)
    state = _state(scales, [50, 50, 50],
                   _image_layer(fixed_url), _image_layer(moving_url))
    monkeypatch.setattr(ngregister, "viewer", types.SimpleNamespace(state=state))

    correction = ngregister.refine_registration(size_voxels=60, apply=False)
    _assert_cancels_shift(correction, shift)


def test_refine_sharded_zarr3_recovers_shift(monkeypatch, tmp_path):
    scales = [4, 4, 8]
    shift = np.array([2.0, 1.0, -1.0])
    vol = _texture()
    moving = ndi.shift(vol, shift, order=1, mode="reflect")

    # Stored [x, y, z] with matching dimension labels -> identity transform.
    fixed_url = _write_zarr3_sharded(tmp_path / "fixed", vol, ["x", "y", "z"])
    moving_url = _write_zarr3_sharded(tmp_path / "moving", moving, ["x", "y", "z"])
    state = _state(scales, [50, 50, 50],
                   _image_layer(fixed_url), _image_layer(moving_url))
    monkeypatch.setattr(ngregister, "viewer", types.SimpleNamespace(state=state))

    correction = ngregister.refine_registration(size_voxels=60, apply=False)
    _assert_cancels_shift(correction, shift)


def test_refine_zarr3_reversed_axes(monkeypatch, tmp_path):
    """Moving volume stored in z,y,x order with a permuting layer transform.

    This is the axis-ordering stress test: TensorStore returns the array in its
    native [z, y, x] order, the transform maps that local order to global x,y,z,
    and the refinement must still recover the shift.
    """
    scales = [4, 4, 8]
    shift = np.array([3.0, -2.0, 1.0])                # in global x,y,z voxels
    vol = _texture()
    moving = ndi.shift(vol, shift, order=1, mode="reflect")

    dims = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=scales)
    fixed_url = _write_precomputed(tmp_path / "fixed", vol, scales)
    # Store moving transposed to [z, y, x]; transform reverses local -> global.
    moving_url = _write_zarr3_sharded(
        tmp_path / "moving", np.transpose(moving, (2, 1, 0)), ["z", "y", "x"])
    reverse = [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]  # local z,y,x -> global x,y,z

    state = _state(scales, [50, 50, 50],
                   _image_layer(fixed_url),
                   _image_layer(moving_url, matrix=reverse, dims=dims))
    monkeypatch.setattr(ngregister, "viewer", types.SimpleNamespace(state=state))

    correction = ngregister.refine_registration(size_voxels=60, apply=False)
    _assert_cancels_shift(correction, shift)


def test_refine_ome_coarse_resolution_recovers_shift(monkeypatch, tmp_path):
    """Source stored at a coarser resolution than the global frame (issue #5).

    The OME level-0 voxel is 8 nm while the viewer frame is 4 nm, so the layer
    transform maps the source's *physical* position (in 4 nm voxel units), not
    the raw array index. The landmark sits where treating the matrix as an
    index map would clamp the box to nothing and raise "does not overlap"; the
    fix must instead read the right cutout and recover the shift.
    """
    global_scale = [4.0, 4.0, 4.0]                     # nm per global voxel
    native_scale = [8.0, 8.0, 8.0]                     # nm per source voxel
    native_translation = [4.0, 4.0, 4.0]               # nm
    ratio = native_scale[0] / global_scale[0]          # 2 global voxels / source voxel
    shift_native = np.array([3.0, -2.0, 1.0])          # in source voxels
    vol = _texture()
    moving = ndi.shift(vol, shift_native, order=1, mode="reflect")

    fixed_url = _write_ome_zarr3_scaled(
        tmp_path / "fixed", vol, native_scale, native_translation)
    moving_url = _write_ome_zarr3_scaled(
        tmp_path / "moving", moving, native_scale, native_translation)
    # global voxel near the far edge: index-as-global would fall out of bounds.
    landmark = [150.0, 150.0, 150.0]
    state = _state(global_scale, landmark,
                   _image_layer(fixed_url), _image_layer(moving_url))
    monkeypatch.setattr(ngregister, "viewer", types.SimpleNamespace(state=state))

    correction = ngregister.refine_registration(size_voxels=60, apply=False)
    # The shift is reported in global voxels (source voxels x ratio).
    _assert_cancels_shift(correction, shift_native * ratio)


def test_refine_ome_multiscale_group(monkeypatch, tmp_path):
    scales = [4, 4, 8]
    shift = np.array([2.0, -1.0, 1.0])
    vol = _texture()
    moving = ndi.shift(vol, shift, order=1, mode="reflect")

    fixed_url = _write_ome_zarr3_group(tmp_path / "fixed", vol, scales)
    moving_url = _write_ome_zarr3_group(tmp_path / "moving", moving, scales)
    state = _state(scales, [50, 50, 50],
                   _image_layer(fixed_url), _image_layer(moving_url))
    monkeypatch.setattr(ngregister, "viewer", types.SimpleNamespace(state=state))

    correction = ngregister.refine_registration(size_voxels=60, apply=False)
    _assert_cancels_shift(correction, shift)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_split_source_url_handles_both_encodings():
    assert ngregister.split_source_url("precomputed://gs://b/p") == (
        "precomputed", "gs://b/p")
    assert ngregister.split_source_url("zarr3://https://h/d.zarr") == (
        "zarr3", "https://h/d.zarr")
    assert ngregister.split_source_url("https://h/d.ome.zarr|zarr3:") == (
        "zarr3", "https://h/d.ome.zarr")
    # bare path becomes a file:// kvstore
    assert ngregister.split_source_url("precomputed:///data/vol")[1] == "file:///data/vol"


def test_source_url_to_spec_selects_driver_and_scale():
    spec = ngregister.source_url_to_spec("precomputed://gs://b/p", mip=2)
    assert spec == {"driver": "neuroglancer_precomputed",
                    "kvstore": "gs://b/p", "scale_index": 2}
    spec = ngregister.source_url_to_spec("zarr3://https://h/d.zarr")
    assert spec["driver"] == "zarr3" and spec["kvstore"] == "https://h/d.zarr"


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
