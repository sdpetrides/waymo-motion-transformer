import os
import time
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

print("GPUs Available: ", tf.config.list_physical_devices("GPU"))
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

from waymo_open_dataset.metrics.python import config_util_py as config_util


from data.dataset import load_dataset, parse_dataset, parse_example_masked
from models.naive import NaiveModel
from metrics import default_metrics_config, MotionMetrics
from train import train_step


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs (default: 2)", default=2
    )
    parser.add_argument("-b", "--batch_size", help="Batch size (default: 2)", default=2)
    parser.add_argument(
        "-r", "--tf_records", help="Number of TFRecords to use (default: 2)", default=2
    )
    args = parser.parse_args()
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    tf_records = int(args.tf_records)
    if tf_records == 0 or tf_records > 1000:
        tf_records = 1

    dataset = load_dataset(tfrecords=tf_records)
    model = NaiveModel(
        num_agents_per_scenario=128, num_state_steps=11, num_future_steps=80
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-2)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics_config = default_metrics_config()
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    dataset = dataset.map(parse_example_masked)
    dataset = dataset.batch(batch_size)

    for epoch in range(epochs):
        print(f"Start of epoch {epoch}")
        # TODO: timing
        # start_time = time.time()

        # Iterate over the batches of the dataset.
        losses = []
        for step, batch in enumerate(dataset):
            loss_value = train_step(
                model, loss_fn, optimizer, batch, metrics_config, motion_metrics
            )

            # Log every 10 batches.
            losses.append(loss_value)
            if step % 10 == 9:
                print(
                    "Avg Training loss for last 10 steps %4d: %12.3f"
                    % (step + 1, float(sum(losses[-10:]) / 10))
                )
                # print("Seen so far: %d samples" % ((step + 1) * batch_size))

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
