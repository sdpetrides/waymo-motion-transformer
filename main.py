import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from waymo_open_dataset.metrics.python import config_util_py as config_util


from data.dataset import load_dataset, parse_dataset, parse_example_masked
from models.naive import NaiveModel
from metrics import default_metrics_config, MotionMetrics
from train import train_step


def main():
    dataset = load_dataset(tfrecords=2)
    model = NaiveModel(
        num_agents_per_scenario=128, num_state_steps=11, num_future_steps=80
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics_config = default_metrics_config()
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    batch_size = 32
    dataset = dataset.map(parse_example_masked)
    dataset = dataset.batch(batch_size)

    epochs = 2
    num_batches_per_epoch = 10

    for epoch in range(epochs):
        print(f"Start of epoch {epoch}")
        # TODO: timing
        # start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, batch in enumerate(dataset):
            loss_value = train_step(
                model, loss_fn, optimizer, batch, metrics_config, motion_metrics
            )

            # Log every 10 batches.
            if step % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

            if step >= num_batches_per_epoch:
                break

        # TODO: Deal with metrics
        # Display metrics at the end of each epoch.
        train_metric_values = motion_metrics.result()
        for i, m in enumerate(
            ["min_ade", "min_fde", "miss_rate", "overlap_rate", "map"]
        ):
            for j, n in enumerate(metric_names):
                print("{}/{}: {}".format(m, n, train_metric_values[i, j]))


if __name__ == "__main__":
    main()
