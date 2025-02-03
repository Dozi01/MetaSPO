import os
import time
from datetime import timedelta
from typing import Optional
import json
from .methods import *
from .utils import get_pacific_time, create_logger
from .language_model import BaseModel, OptimizationModel
from .taskmanager import TaskManager
from .methods.node import Node

OPTIMIZE_METHOD_DICT = {
    "metaspo": MetaSPO,
    "outer_loop": MetaSPO,
    "unseen_generalization": MetaSPO,  # for unseen generalization, dummy method MetaSPO is used
    "metaspo_ape": MetaSPOAPE,
    "ape": APE,
    "apo": APO,
}


class Runner:
    def __init__(
        self,
        log_dir: str,
        meta_train_tasks: list[str],
        meta_test_tasks: list[str],
        task_setting: dict,
        base_model_setting: dict,
        optim_model_setting: dict,
        search_setting: dict,
        init_system_prompt_path: Optional[str],
        **kwargs,
    ) -> None:

        # Load initial system prompt from file
        init_system_prompt = self.get_system_prompt(init_system_prompt_path)

        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}'

        self.log_dir = os.path.join(log_dir, exp_name)
        self.logger = create_logger(self.log_dir)

        self.task_manager = TaskManager(meta_train_tasks, meta_test_tasks, task_setting)

        # Initialize base model and optimization model
        self.base_model = BaseModel(base_model_setting, self.logger)
        self.optim_model = OptimizationModel(optim_model_setting, self.logger)

        # Initialize optimization method
        self.optim_method = OPTIMIZE_METHOD_DICT[search_setting["method"]](
            task_manager=self.task_manager,
            base_model=self.base_model,
            optim_model=self.optim_model,
            initial_system_prompt=init_system_prompt,
            log_dir=self.log_dir,
            logger=self.logger,
            **search_setting,
        )

        self.logger.info(f"base_model_setting : {base_model_setting}")
        self.logger.info(f"optim_model_setting : {optim_model_setting}")
        self.logger.info(f"search_setting : {search_setting}")
        self.logger.info(f"task_setting : {task_setting}")
        self.logger.info(f"meta_train_tasks : {meta_train_tasks}")
        self.logger.info(f"meta_test_tasks : {meta_test_tasks}")
        self.logger.info(f"init_system_prompt_path : {init_system_prompt_path}")
        self.logger.info(f"init_system_prompt : {init_system_prompt}")

    def meta_train(self):
        """
        Start searching from initial prompt
        """
        start_time = time.time()
        self.optimized_system_prompt = self.optim_method.train()
        meta_train_time = time.time()
        end_time = time.time()

        meta_train_time = str(timedelta(seconds=meta_train_time - start_time)).split(".")[0]
        self.logger.info(f"optimized_system_prompt :\n{self.optimized_system_prompt}")
        self.logger.info(f"\nDone!Meta Train time: {meta_train_time}")
        exe_time = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        self.logger.info(f"\nDone!Excution time: {exe_time}")
        return

    def get_system_prompt(self, file_path):
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                file_name = os.path.basename(file_path)
                if "bilevel_nodes" in file_name:
                    if isinstance(data[-1][-1], dict) and "system_prompt" in data[-1][-1]:
                        system_prompt = data[-1][-1]["system_prompt"]
                    elif isinstance(data[-1][-1][-1], dict) and "system_prompt" in data[-1][-1][-1]:
                        system_prompt = data[-1][-1][-1]["system_prompt"]
                else:
                    system_prompt = data["prompt"]

                return system_prompt
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
