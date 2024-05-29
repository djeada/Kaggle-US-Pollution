# logging_config.py

import logging.config

import yaml


def setup_logging(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config["logging"])
