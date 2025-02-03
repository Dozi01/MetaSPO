import os
import logging
from glob import glob
from datetime import datetime
import pytz
import openai
import yaml
import argparse


openai.log = logging.getLogger("openai")
openai.log.setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


class HTTPFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("HTTP")


def get_pacific_time():
    current_time = datetime.now()
    pacific = pytz.timezone("Asia/Seoul")
    pacific_time = current_time.astimezone(pacific)
    return pacific_time


def create_logger(logging_dir, name="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    logging_dir = os.path.join(logging_dir, name)
    num = len(glob(logging_dir + "*"))

    logging_dir += "-" + f"{num:03d}" + ".log"
    http_filter = HTTPFilter()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}")],
    )
    logger = logging.getLogger("prompt optimization agent")
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    for handler in logging.getLogger().handlers:
        handler.addFilter(http_filter)
    return logger


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_API_KEY(config):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    config["base_model_setting"]["api_key"] = openai_api_key
    config["optim_model_setting"]["api_key"] = openai_api_key
    return config


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--base_config_dir", type=str, default="./configs/base_config.yaml")
    parser.add_argument("--init_system_prompt_path", type=str, default="./prompts/default.json")

    parser.add_argument("--method", type=str, required=True)

    parser.add_argument("--log_dir", type=str, required=True)

    parser.add_argument("--base_model_type", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config_dir)
    base_config = load_config(args.base_config_dir)
    config.update(base_config)
    config["base_model_setting"]["model_type"] = args.base_model_type
    config["base_model_setting"]["model_name"] = args.base_model_name

    config["search_setting"]["method"] = args.method

    config["init_system_prompt_path"] = args.init_system_prompt_path
    config["log_dir"] = args.log_dir

    return config
