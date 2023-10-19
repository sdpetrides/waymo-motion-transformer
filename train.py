import tensorflow as tf


def train_step(model, loss_fn, optimizer, inputs, metrics_config, motion_metrics):
    with tf.GradientTape() as tape:
        # [batch_size, num_agents, D]
        states = inputs["all_states_masked"]
        is_valid = inputs["gt_future_is_valid"]
        slice_index = inputs["masked_index"]

        # Predict. [batch_size, num_agents, V].
        pred_trajectory = model(states, is_valid, training=True)

        # Set training target.
        prediction_start = metrics_config.track_history_samples + 1

        # [batch_size, num_agents, V]
        gt_trajectory = tf.transpose(inputs["gt_future_states"], perm=[0, 2, 1, 3])
        gt_targets = tf.gather(gt_trajectory, indices=slice_index, axis=1, batch_dims=1)

        # TODO: deal with validity
        # [batch_size, num_agents, steps]
        gt_is_valid = inputs["gt_future_is_valid"]
        # # [batch_size, num_agents, steps]
        # weights = tf.cast(
        #     inputs["gt_future_is_valid"][..., prediction_start:], tf.float32
        # ) * tf.cast(inputs["tracks_to_predict"][..., tf.newaxis], tf.float32)

        loss_value = loss_fn(gt_targets, pred_trajectory)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # [batch_size, num_agents, steps, 2] ->
    # [batch_size, num_agents, 1, 1, steps, 2].
    # The added dimensions are top_k = 1, num_agents_per_joint_prediction = 1.
    pred_trajectory = pred_trajectory[:, :, tf.newaxis, tf.newaxis]

    # Fake the score since this model does not generate any score per predicted
    # trajectory.
    pred_score = tf.ones(shape=tf.shape(pred_trajectory)[:3])

    # [batch_size, num_agents].
    object_type = inputs["object_type"]

    # [batch_size, num_agents].
    batch_size = tf.shape(inputs["tracks_to_predict"])[0]
    num_samples = tf.shape(inputs["tracks_to_predict"])[1]

    pred_gt_indices = tf.range(num_samples, dtype=tf.int64)
    # [batch_size, num_agents, 1].
    pred_gt_indices = tf.tile(
        pred_gt_indices[tf.newaxis, :, tf.newaxis], (batch_size, 1, 1)
    )
    # [batch_size, num_agents, 1].
    pred_gt_indices_mask = inputs["tracks_to_predict"][..., tf.newaxis]

    # motion_metrics.update_state(
    #     pred_trajectory,
    #     pred_score,
    #     gt_trajectory,
    #     gt_is_valid,
    #     pred_gt_indices,
    #     pred_gt_indices_mask,
    #     object_type,
    # )

    return loss_value
