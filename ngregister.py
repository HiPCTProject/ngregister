import argparse
import webbrowser

import neuroglancer
import neuroglancer.cli
import numpy as np
from scipy.spatial.transform import Rotation
import datetime
import atexit

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
def apply_transform_to_layer(applied_transform_matrix, chain_backwards: bool = False):
    global viewer
    with viewer.txn() as v:
        if v.layers[v.selectedLayer.layer].layer.type == "annotation":
            return
        current_transform = v.layers[v.selectedLayer.layer].layer.source[0].transform
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
        current_source_url = v.layers[v.selectedLayer.layer].layer.source[0].url
        v.layers[v.selectedLayer.layer].layer.source[0] = neuroglancer.LayerDataSource({"url": current_source_url, "transform": new_transform.to_json()})

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