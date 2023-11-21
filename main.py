import os
import sys
import time
import argparse
import logging
import random

import datetime as dt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np

import wandb

run = dt.datetime.now().strftime("%Y-%m-%d_%H:%M")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(f"{run}.log")],
)
logger = logging.getLogger(__name__)

seed = 16

tf.random.set_seed(seed)

logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

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
    N_rg_encoder = 2
    N_obj_encoder = 2
    N_decoder = 4

    logger.info(f"use_performers: {use_performers}")
    logger.info(f"use_center: {use_center}")

    metrics_config = default_metrics_config()
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    buffer_size = tf_records * 1000
    global_batch_size = batch_size
    logger.info(f"global batch size {global_batch_size}")

    learning_rate = 2e-4
    weight_decay = 0.9999

    parse_example = parser_factory(use_center=use_center)
    dataset = load_dataset(tfrecords=tf_records)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(global_batch_size)

    model = NaiveModel(
        num_agents_per_scenario=128,
        num_state_steps=11,
        num_future_steps=80,
        N_rg_encoder=N_rg_encoder,
        N_obj_encoder=N_obj_encoder,
        N_decoder=N_decoder,
        use_performers=use_performers,
    )
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.build(input_shape=[(None, 11, 1024), (None, 30000, 11), (None, 40, 1024)])
    model.summary(expand_nested=True, show_trainable=True)

    wandb.init(
        project="waymo-motion",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "tf_records": tf_records,
            "batch_size": batch_size,
            "use_performers": use_performers,
            "use_center": use_center,
            "random_seed": seed,
            "N_rg_encoder": N_rg_encoder,
            "N_obj_encoder": N_obj_encoder,
            "N_decoder": N_decoder,
        },
    )

    epoch_losses = []
    for epoch in range(epochs):
        logger.info(f"Start of epoch {epoch}")

        # Iterate over the batches of the dataset.
        losses = []
        for step, batch in enumerate(dataset):
            # start_time = time.time()
            loss_value = train_step(
                model, loss_fn, optimizer, batch, metrics_config, motion_metrics
            )
            # logger.info(step, loss_value, time.time() - start_time)
            wandb.log({"loss": loss_value, "learning_rate": optimizer.learning_rate})
            losses.append(loss_value)

            # Log every 10 batches.
            if step % 10 == 9:
                logger.info(
                    "Avg Training loss for last 10 steps %4d: %12.3f"
                    % (step + 1, float(sum(losses[-10:]) / 10))
                )
                # print("Seen so far: %d samples" % ((step + 1) * batch_size))

        epoch_loss = sum(losses) / len(losses)
        wandb.log({"epoch_loss": epoch_loss})
        logger.info(f"Epoch {epoch}: avg loss: {epoch_loss}")
        epoch_losses.append(epoch_loss)

    wandb.finish()
    file_path = f"{run}_loss.npy"
    np.save(file_path, np.array(epoch_losses))

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
