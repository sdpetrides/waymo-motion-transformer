import os
import time
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

tf.random.set_seed(16)

print("GPUs Available: ", tf.config.list_physical_devices("GPU"))
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

from waymo_open_dataset.metrics.python import config_util_py as config_util


from data.dataset import load_dataset, parser_factory
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
    parser.add_argument(
        "-p", "--use_performers", help="Use performers (default: True)", default="True"
    )
    parser.add_argument(
        "-c",
        "--use_center",
        help="Center everything around AV (default: True)",
        default="True",
    )
    args = parser.parse_args()
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    tf_records = int(args.tf_records)
    if tf_records == 0 or tf_records > 1000:
        tf_records = 1
    use_performers = (
        True if args.use_performers.lower() in ("true", "t", "1") else False
    )
    use_center = True if args.use_center.lower() in ("true", "t", "1") else False

    print(f"use_performers: {use_performers}")
    print(f"use_center: {use_center}")

    model = NaiveModel(
        num_agents_per_scenario=128,
        num_state_steps=11,
        num_future_steps=80,
        use_performers=use_performers,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics_config = default_metrics_config()
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    parse_example = parser_factory(use_center=use_center)
    dataset = load_dataset(tfrecords=tf_records)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(tf_records * 1000)
    dataset = dataset.batch(batch_size)

    for epoch in range(epochs):
        print(f"Start of epoch {epoch}")

        # Iterate over the batches of the dataset.
        losses = []
        for step, batch in enumerate(dataset):
            # start_time = time.time()
            loss_value = train_step(
                model, loss_fn, optimizer, batch, metrics_config, motion_metrics
            )
            # print(step, loss_value, time.time() - start_time)

            # Log every 10 batches.
            losses.append(loss_value)
            if step % 10 == 9:
                print(
                    "Avg Training loss for last 10 steps %4d: %12.3f"
                    % (step + 1, float(sum(losses[-10:]) / 10))
                )
                # print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # # TODO: Deal with metrics
        # # Display metrics at the end of each epoch.
        # train_metric_values = motion_metrics.result()
        # for i, m in enumerate(
        #     ["min_ade", "min_fde", "miss_rate", "overlap_rate", "map"]
        # ):
        #     for j, n in enumerate(metric_names):
        #         print("{}/{}: {}".format(m, n, train_metric_values[i, j]))


if __name__ == "__main__":
    main()
