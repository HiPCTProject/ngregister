# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ngregister starts a local Neuroglancer instance with custom keyboard shortcuts for manually
registering (translating, rotating, flipping) one volume layer onto another, plus an automated
local refinement step that uses TensorStore + SimpleITK to fine-tune the alignment around a
landmark. It is a single script, `ngregister.py`, driven as an interactive Python session.

## Running

```bash
# blank state
python -i ngregister.py

# resume from a saved JSON state file or a Neuroglancer URL
python -i ngregister.py --url "https://neuroglancer-demo.appspot.com/#!..."
```

The script opens a browser tab to the viewer and drops you into the Python REPL (`-i`). On exit
it prints the current Neuroglancer URL (via an `atexit` hook) so the registered state can be
recovered.

Dependencies (`requirements.txt`): `neuroglancer`, `numpy`, `scipy`, plus `tensorstore` and
`SimpleITK` for the refinement step. To support newer Neuroglancer source URLs (e.g.
`gs://bucket/data.ome.zarr|zarr:`), install the dev build:
`pip install git+https://github.com/google/neuroglancer.git` (needs a `node` + `Python.h` build env).

There is no build step, lint config, or test suite.

## Architecture

A single module-level `neuroglancer.Viewer()` singleton (`viewer`) holds all state and is mutated
through `viewer.txn()` (state) and `viewer.config_state.txn()` (key bindings).

The central abstraction is `apply_transform_to_layer(matrix, chain_backwards=False)`: every
registration gesture computes a 4x4 homogeneous transform and passes it here, where it is composed
with the selected layer's existing `source[0].transform` and written back. Annotation layers are
ignored. `chain_backwards` switches between left- and right-multiplication of the existing transform.

Gesture builders:
- `compute_rotation_around_main_axis(s, axis_name)` and `flip_main_axis(s, axis_name)` build their
  matrices as a translate-to-view-center / rotate-or-flip / translate-back sandwich, so rotations
  and flips happen about the current view center around absolute x/y/z axes. Rotation step is
  `angle_step_in_degrees` (3 degrees). A leading `-` in `axis_name` reverses direction.
- Translation gestures build a pure-translation matrix from cursor/center/landmark voxel coordinates.

Special annotation layers (created on demand, named with double underscores):
- `__LANDMARK__` holds a single point used by the place-landmark / translate-to-landmark workflow.
- `__HISTORY__` plus the in-memory `history` dict snapshot viewer state on each rotation
  (`save_state_in_history` / `load_state_from_history`).

Actions and key bindings are wired at the bottom of the `__main__` block: each function is registered
with `viewer.actions.add('action-name', fn)` and bound to a key in the `viewer.config_state.txn()`
block (e.g. `s.input_event_bindings.viewer['keyt'] = 'translate-layer-with-cursor'`).

## Refinement step

`refine_registration()` (callable from the REPL, defined in the same `ngregister.py`) fetches a small
subvolume around the landmark from a reference and a moving layer with TensorStore, runs a SimpleITK
registration, and composes the correction back onto the moving layer via the same
`apply_transform_to_layer`. Two parameters tune it: `metric` ('correlation' default, 'meansquares',
or 'mattes') and `model` ('rigid' default, 'similarity', or 'affine'). The defaults suit two scans of
the *same* specimen at different resolutions (the common case): correlation/meansquares give a far
stronger gradient than Mattes MI on a small, blurry overview box (MI is for cross-modal pairs and is
near-flat here), and the rigid/similarity models cannot represent shear, so the optimizer cannot
spend ill-constrained linear DOF on spurious shear/anisotropic scale the way an unconstrained 12-DOF
affine does. `mip` selects the multiscale level fetched from each source (0 = full resolution);
`fixed_mip` / `moving_mip` override it per layer, so a coarse overview and a fine VOI can be compared
at whichever levels give a comparable working resolution. Level selection flows through `_open_source`
(resolves the OME group's level path) and `_native_voxel_geometry` (reads that level's
`coordinateTransformations`), so each cutout is framed for its own level and the two still share a
frame regardless of the levels chosen.

The refinement code is built around explicit affine bookkeeping in one shared frame because the three
libraries disagree on axis order:
- Neuroglancer positions are global/output **voxel** coordinates ordered like `viewer.dimensions`; a
  layer's `source[0].transform.matrix` is a 3x4 affine mapping the source's *intrinsic* coordinate ->
  global voxel. The intrinsic coordinate is **not** the raw array index: it is the source's physical
  position expressed in global-voxel units, so the matrix is near-identity even when the source is
  stored at a different resolution than the global frame. `_native_voxel_geometry` recovers the
  per-axis `(ratio, offset)` that turns an array index into that intrinsic coordinate from the
  source's resolution metadata (OME `coordinateTransformations` for zarr via `_ome_multiscale`,
  precomputed `resolution`); `fetch_layer_image` composes it with the transform into the full
  array-index -> global-voxel affine used for both the box clamp and the SimpleITK geometry. Without
  this an OME overview level (e.g. 20 um data in a 4 um frame) maps the box ~5x too far and misses
  the array bounds (issue #5). Falls back to `(1, 0)` when no resolution metadata is found.
- TensorStore opens the source lazily (precomputed, zarr, sharded zarr3, n5) and only the small
  cutout is read; the returned array's axes are in the source's stored order (= the transform's
  column order). `source_url_to_spec` builds the open spec (driver + kvstore + `scale_index`) purely;
  `_open_source` opens it and, if a zarr source is an OME multiscale *group*, resolves the level path
  from the metadata on a retry. `_spatial_axis_order` uses TensorStore dimension labels to drop the
  channel axis and confirm spatial axes (no hardcoded x-first assumption).
- SimpleITK images are indexed `(x, y, z)` but `Get/SetImageFromArray` use reversed `[z, y, x]`;
  geometry is physical via origin/spacing/direction.

Pipeline (`fetch_layer_image` -> `register_pair` -> compose): each cutout is reversed
`[a0,a1,a2]->[a2,a1,a0]` for SimpleITK and placed in global **physical** space (= global voxel x
`dimensions.scales` converted to metres via `_unit_to_meters`, then normalized to the finest axis so
ITK sees O(1) spacings; the common factor cancels in the round-trip) by
setting the image origin/spacing/direction from the layer affine (`polar_decompose` splits its
linear part into an orthonormal direction + per-axis spacing; assumes no shear, which is what
ngregister's gestures produce). Both images thus share a frame, so registration starts at identity.
`register_pair` is staged (translation, then the chosen `model` via `_model_transform`) with a gentle
`[2,1]` pyramid and a RegularStepGradientDescent optimizer — an aggressive pyramid or a from-scratch
12-DOF affine diverges on small subvolumes. The second-stage transform is **centered on the image**
(`TransformContinuousIndexToPhysicalPoint` of the size midpoint): the cutout sits thousands of voxels
from the physical origin, so a transform centered at 0 would make `SetOptimizerScalesFromPhysicalShift`
freeze the linear DOF and only the translation would move. The SimpleITK transform `T` maps
fixed->moving physical points; the correction applied to the moving layer is `inv(T)`, converted back
to global-voxel units. `refine_registration` reports the correction as the actual displacement it
induces at the landmark and over the box corners — not the homogeneous translation column, which is
large for a rotation about a far-from-origin center even when the box barely moves.

Layer roles come from name prefixes: `mark-layer-moving` / `mark-layer-reference` (keys
`alt+m` / `alt+r`) rename the active layer with a `mov::` / `ref::` prefix (constants
`MOVING_PREFIX` / `REFERENCE_PREFIX`). `resolve_roles` reads those prefixes, else falls back
to active=moving / other-image-layer=fixed, else raises.

## Adding a new registration gesture

1. Write a handler `def my_action(s):` that builds a 4x4 matrix and calls `apply_transform_to_layer(...)`
   (optionally `save_state_in_history()`).
2. Register it: `viewer.actions.add('my-action', my_action)`.
3. Bind a key inside the `config_state.txn()` block: `s.input_event_bindings.viewer['keyX'] = 'my-action'`.

The `s` argument is the Neuroglancer action state; useful fields include `s.mouse_voxel_coordinates`
and `s.viewer_state.voxel_coordinates` (view center). Always guard against `mouse_voxel_coordinates`
being `None`.
