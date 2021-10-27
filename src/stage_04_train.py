from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import matplotlib.pyplot as plt
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

def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    performance_graph_dir = os.path.join(artifacts_dir, artifacts["PERFORMANCE_GRAPH"])

    create_directory([train_model_dir_path, performance_graph_dir])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL_NAME"])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path  = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=params["EPOCHS"], 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"{performance_graph_dir}{'/accuracy.png'}")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"{performance_graph_dir}{'/loss.png'}")


    logging.info(f"training completed")

    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directory([trained_model_dir])

    with open(config_path) as yaml_file:
        content = yaml.safe_load(yaml_file)

    model_file_path = get_unique_path_to_save_model(trained_model_dir)
    
    content['new_save_model_path']['SAVE_MODEL_DIR'] = model_file_path

    with open(config_path, 'w') as f:
        yaml.dump(content, f)

    model.save(model_file_path)
    logging.info(f"trained model is saved at: {model_file_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage four started")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage four completed! training completed and model is saved >>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e