import yaml
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.callbacks import configuration_settings
from model.trainer import MSTCN_AutoEncoder


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file) 

dataset = np.load(Path(config['output']['tensor_dir']) / "tensors.npz")

parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
parser.add_argument(
    "--mode",
    required=True,
    choices=['train', 'extract', 'tune', 'predict'], 
    help="Select which process within the pipeline to be executed")
args = parser.parse_args()


def main():
    model = MSTCN_AutoEncoder(
        features=dataset['train_ds'].shape[-1],
        latent_dim=config['latent_dim'],
        train_ds=dataset["train_ds"],
        val_ds=dataset["val_ds"],
        test_ds=dataset["test_ds"])

    if args.mode == "train":
        model.train(
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            callbacks=configuration_settings(config))

    elif args.mode == "extract":
        model.load_best_weights(config['model']['checkpoints'])

        train_anomaly_scores = model.calculate_anomaly_scores(dataset['train_ds'])
        val_anomaly_scores = model.calculate_anomaly_scores(dataset['val_ds'])
        test_anomaly_scores = model.calculate_anomaly_scores(dataset['test_ds'])

        save_path = Path(config['model']['anomaly_scores'])
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path / "scores.npz",
            train_ds=train_anomaly_scores,
            val_ds=val_anomaly_scores,
            test_ds=test_anomaly_scores)

    elif args.mode == "tune":
        scores =  np.load(Path(config['model']['anomaly_scores']) / "scores.npz")

        best_param, _ = model.tune_hyperparameters(
            reference_scores=scores['train'],
            target_scores=scores['val'],
            ground_truth=dataset['val_labels']
            percentile_grid=[95, 99, 99.5],
            min_features_grid=dataset['train'].shape[-1])

        config['model']['percentile'] = best_param['percentile']

    elif args.mode == "predict":
        scores =  np.load(Path(config['model']['anomaly_scores']) / "scores.npz")

        detection_results = model.detect_anomalies(
            referece_scores=scores['train'],
            target_scores=scores['test'],
            percentile=config['model']['percentile'])

        model.evaluate(
            anomaly_scores=scores['test'],
            ground_truth=dataset["test_labels"],
            detection_results=detection_results,
            minimum_abnormal_features=config['minimum_abnormal_features'])


if __name__ == "__main__":
    main()