import os
import time
from datetime import timedelta
import json
from .methods import *
from .utils import get_pacific_time, create_logger
from .language_model import BaseModel, OptimizationModel
from .taskmanager import TaskManager

OPTIMIZE_METHOD_DICT = {
    "metaspo": MetaSPO,
    "metaspo_ape": MetaSPOAPE,
    "outer_loop": MetaSPO,
    "unseen_generalization": MetaSPO,  # for unseen generalization, dummy method MetaSPO is used
    "test_time_adaptation": ProTeGi,  # for test time adaptation
    "ape": APE,
    "protegi": ProTeGi,
}


class Runner:
    def __init__(self, args):

        # Load initial system prompt from file
        self.init_system_prompt = self.get_system_prompt(args.init_system_prompt_path)
        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}'

        self.log_dir = os.path.join(args.log_dir, exp_name)
        self.logger = create_logger(self.log_dir)

        search_setting, base_model_setting, optim_model_setting, task_setting = self.parse_args(args)

        self.task_manager = TaskManager(args.meta_train_tasks, args.meta_test_tasks, task_setting)

        # Initialize base model and optimization model
        self.base_model = BaseModel(base_model_setting, self.logger)
        self.optim_model = OptimizationModel(optim_model_setting, self.logger)

        # Initialize optimization method
        self.optim_method = OPTIMIZE_METHOD_DICT[search_setting["method"]](
            task_manager=self.task_manager,
            base_model=self.base_model,
            optim_model=self.optim_model,
            initial_system_prompt=self.init_system_prompt,
            log_dir=self.log_dir,
            logger=self.logger,
            **search_setting,
        )

        self.logger.info(f"base_model_setting : {base_model_setting}")
        self.logger.info(f"optim_model_setting : {optim_model_setting}")
        self.logger.info(f"search_setting : {search_setting}")
        self.logger.info(f"task_setting : {task_setting}")
        self.logger.info(f"meta_train_tasks : {args.meta_train_tasks}")
        self.logger.info(f"meta_test_tasks : {args.meta_test_tasks}")
        self.logger.info(f"init_system_prompt_path : {args.init_system_prompt_path}")
        self.logger.info(f"init_system_prompt : {self.init_system_prompt}")

    def meta_train(self):
        """
        Start searching from initial prompt
        """
        start_time = time.time()
        self.optim_method.train()
        end_time = time.time()

        exe_time = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        self.logger.info(f"\nExcution time: {exe_time}")
        return

    def get_system_prompt(self, file_path):
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                file_name = os.path.basename(file_path)
                if "bilevel_nodes" in file_name:
                    system_prompt = data['optimized_system_prompt']
                else:
                    system_prompt = data["prompt"]

                return system_prompt
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

    def parse_args(self, args):
        search_setting = {
            "method": args.method,
            "iteration": args.iteration,
            "num_system_candidate": args.num_system_candidate,
            "num_user_candidate": args.num_user_candidate,
            "user_top_k": args.user_top_k,
        }

        base_model_setting = {
            "model_type": args.base_model_type,
            "model_name": args.base_model_name,
            "temperature": args.base_model_temperature,
            "api_key": args.openai_api_key,
        }

        optim_model_setting = {
            "model_type": args.optim_model_type,
            "model_name": args.optim_model_name,
            "temperature": args.optim_model_temperature,
            "api_key": args.openai_api_key,
        }

        task_setting = {
            "train_size": args.train_size,
            "test_size": args.test_size,
            "seed": args.seed,
            "data_dir": args.dataset_dir,
        }

        return search_setting, base_model_setting, optim_model_setting, task_setting
