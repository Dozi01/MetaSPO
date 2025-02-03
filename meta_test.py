from src.utils import config, get_API_KEY
from src.analyser import Analyser

if __name__ == "__main__":
    configs = config()
    configs = get_API_KEY(configs)

    analyser = Analyser(**configs)
    analyser.meta_test()
