import tensorflow as tf


def train_step(model, loss_fn, optimizer, inputs, metrics_config, motion_metrics):
    """"""
    with tf.GradientTape() as tape:
        road_graph = inputs["road_graph"]  # B, T, V_rg
        states = inputs["gt_future_states"]  # B, obj, T, V_obj

        # Swap T and obj dims, merge with V to get H
        states_gt = model.encode(states)  # B, T, H

        model_inputs = states_gt[:, :11, :]  # only use past and present

        # Predict 80 steps
        model_outputs = model(model_inputs, road_graph, training=True)

        loss_value = 0
        for i in [2, 4, 8, 16, 32]:
            gt_state = states_gt[:, 11 + i - 1, :]
            pred_state = model_outputs[:, i, :]
            latent_loss_pred = model.latent_loss_layer(pred_state)
            latent_loss_gt = model.latent_loss_layer(gt_state)
            # Weighted loss value
            loss_value += (1 / (i + 1)) * loss_fn(latent_loss_gt, latent_loss_pred)
        # print(f"Latent loss pred: {latent_loss_pred.shape}")
        # print(f"Latent loss gt: {latent_loss_gt.shape}")

        # Set training target.
        # prediction_start = metrics_config.track_history_samples + 1

        # # [batch_size, num_agents, V]
        # gt_trajectory = tf.transpose(inputs["gt_future_states"], perm=[0, 2, 1, 3])
        # gt_targets = tf.gather(gt_trajectory, indices=slice_index, axis=1, batch_dims=1)

        # TODO: deal with validity
        # # [batch_size, num_agents, steps]
        # gt_is_valid = inputs["gt_future_is_valid"]
        # # [batch_size, num_agents, steps]
        # weights = tf.cast(
        #     inputs["gt_future_is_valid"][..., prediction_start:], tf.float32
        # ) * tf.cast(inputs["tracks_to_predict"][..., tf.newaxis], tf.float32)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # TODO: deal with metrics and trajectories

    # # [batch_size, num_agents, steps, 2] ->
    # # [batch_size, num_agents, 1, 1, steps, 2].
    # # The added dimensions are top_k = 1, num_agents_per_joint_prediction = 1.
    # pred_trajectory = pred_trajectory[:, :, tf.newaxis, tf.newaxis]

    # # Fake the score since this model does not generate any score per predicted
    # # trajectory.
    # pred_score = tf.ones(shape=tf.shape(pred_trajectory)[:3])

    # # [batch_size, num_agents].
    # object_type = inputs["object_type"]

    # # [batch_size, num_agents].
    # batch_size = tf.shape(inputs["tracks_to_predict"])[0]
    # num_samples = tf.shape(inputs["tracks_to_predict"])[1]

    # pred_gt_indices = tf.range(num_samples, dtype=tf.int64)
    # # [batch_size, num_agents, 1].
    # pred_gt_indices = tf.tile(
    #     pred_gt_indices[tf.newaxis, :, tf.newaxis], (batch_size, 1, 1)
    # )
    # # [batch_size, num_agents, 1].
    # pred_gt_indices_mask = inputs["tracks_to_predict"][..., tf.newaxis]

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
