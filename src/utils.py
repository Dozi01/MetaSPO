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

