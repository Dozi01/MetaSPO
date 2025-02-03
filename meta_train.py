from src.utils import config, get_API_KEY
from src.runner import Runner

if __name__ == "__main__":
    configs = config()
    configs = get_API_KEY(configs)

    runner = Runner(**configs)
    runner.meta_train()
