from src.utils.all_utils import read_yaml, create_directory,save_reports
from src.utils.models import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
import logging
import yaml
import pandas as pd

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def evaluate(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    new_save_model_path = config['new_save_model_path']['SAVE_MODEL_DIR']
    SAVE_EVALUATION_DIR = os.path.join(artifacts_dir, artifacts["SAVE_EVALUATION_DIR"])

    create_directory([SAVE_EVALUATION_DIR])

   
    train_generator, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )

    # loading the best perfoming model
    model = tf.keras.models.load_model(new_save_model_path)

    # Getting test accuracy and loss
    test_loss, test_acc = model.evaluate(valid_generator)
    print('Test loss: {} Test Acc: {}'.format(test_loss, test_acc))

    scores = {
        "Test_loss": test_loss,
        "Test_Acc": test_acc
    }
    save_reports(report=scores, report_path=f"{SAVE_EVALUATION_DIR}{'/reports.json'}")



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage five started")
        evaluate(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage five completed! evaluation completed>>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e