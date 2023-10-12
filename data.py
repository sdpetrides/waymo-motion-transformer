import tensorflow as tf


# If you use a custom conversion from Scenario to tf.Example, set the correct
# number of map samples here.
num_map_samples = 30000  # some kind of magic number here

# Example field definition
roadgraph_features = {
    "roadgraph_samples/dir": tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    ),
    "roadgraph_samples/id": tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/type": tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/valid": tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/xyz": tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    ),
}
# Features of other agents.
state_features = {
    "state/id": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    "state/is_sdc": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    "state/tracks_to_predict": tf.io.FixedLenFeature(
        [128], tf.int64, default_value=None
    ),
    "state/current/bbox_yaw": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/height": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/length": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/timestamp_micros": tf.io.FixedLenFeature(
        [128, 1], tf.int64, default_value=None
    ),
    "state/current/valid": tf.io.FixedLenFeature(
        [128, 1], tf.int64, default_value=None
    ),
    "state/current/vel_yaw": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/velocity_x": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/velocity_y": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/width": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/x": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    "state/current/y": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    "state/current/z": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    "state/future/bbox_yaw": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/height": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/length": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/timestamp_micros": tf.io.FixedLenFeature(
        [128, 80], tf.int64, default_value=None
    ),
    "state/future/valid": tf.io.FixedLenFeature(
        [128, 80], tf.int64, default_value=None
    ),
    "state/future/vel_yaw": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/velocity_x": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/velocity_y": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/width": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/x": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    "state/future/y": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    "state/future/z": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    "state/past/bbox_yaw": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/height": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/length": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/timestamp_micros": tf.io.FixedLenFeature(
        [128, 10], tf.int64, default_value=None
    ),
    "state/past/valid": tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    "state/past/vel_yaw": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/velocity_x": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/velocity_y": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/width": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/x": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    "state/past/y": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    "state/past/z": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    "traffic_light_state/current/state": tf.io.FixedLenFeature(
        [1, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/current/valid": tf.io.FixedLenFeature(
        [1, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/current/x": tf.io.FixedLenFeature(
        [1, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/current/y": tf.io.FixedLenFeature(
        [1, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/current/z": tf.io.FixedLenFeature(
        [1, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/past/state": tf.io.FixedLenFeature(
        [10, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/past/valid": tf.io.FixedLenFeature(
        [10, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/past/x": tf.io.FixedLenFeature(
        [10, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/past/y": tf.io.FixedLenFeature(
        [10, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/past/z": tf.io.FixedLenFeature(
        [10, 16], tf.float32, default_value=None
    ),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)


def parse_dataset(value):
    decoded_example = tf.io.parse_single_example(value, features_description)

    past_states = tf.stack(
        [
            decoded_example["state/past/x"],
            decoded_example["state/past/y"],
            decoded_example["state/past/length"],
            decoded_example["state/past/width"],
            decoded_example["state/past/bbox_yaw"],
            decoded_example["state/past/velocity_x"],
            decoded_example["state/past/velocity_y"],
        ],
        -1,
    )

    cur_states = tf.stack(
        [
            decoded_example["state/current/x"],
            decoded_example["state/current/y"],
            decoded_example["state/current/length"],
            decoded_example["state/current/width"],
            decoded_example["state/current/bbox_yaw"],
            decoded_example["state/current/velocity_x"],
            decoded_example["state/current/velocity_y"],
        ],
        -1,
    )

    input_states = tf.concat([past_states, cur_states], 1)[..., :2]

    future_states = tf.stack(
        [
            decoded_example["state/future/x"],
            decoded_example["state/future/y"],
            decoded_example["state/future/length"],
            decoded_example["state/future/width"],
            decoded_example["state/future/bbox_yaw"],
            decoded_example["state/future/velocity_x"],
            decoded_example["state/future/velocity_y"],
        ],
        -1,
    )

    gt_future_states = tf.concat([past_states, cur_states, future_states], 1)

    past_is_valid = decoded_example["state/past/valid"] > 0
    current_is_valid = decoded_example["state/current/valid"] > 0
    future_is_valid = decoded_example["state/future/valid"] > 0
    gt_future_is_valid = tf.concat(
        [past_is_valid, current_is_valid, future_is_valid], 1
    )

    # If a sample was not seen at all in the past, we declare the sample as
    # invalid.
    sample_is_valid = tf.reduce_any(tf.concat([past_is_valid, current_is_valid], 1), 1)

    inputs = {
        "input_states": input_states,
        "gt_future_states": gt_future_states,
        "gt_future_is_valid": gt_future_is_valid,
        "object_type": decoded_example["state/type"],
        "tracks_to_predict": decoded_example["state/tracks_to_predict"] > 0,
        "sample_is_valid": sample_is_valid,
    }
    return inputs


def load_dataset(tfrecords=1000):
    dataset = tf.data.TFRecordDataset(
        [
            f"gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-{i:05d}-of-01000"
            for i in range(tfrecords)
        ],
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    return dataset
