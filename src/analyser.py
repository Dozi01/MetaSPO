from typing import Optional
import json
from .methods import *
from .runner import Runner
from .methods.node import Node
import os


class Analyser(Runner):
    def __init__(
        self,
        log_dir: str,
        meta_train_tasks: list[str],
        meta_test_tasks: list[str],
        task_setting: dict,
        base_model_setting: dict,
        optim_model_setting: dict,
        search_setting: dict,
        init_system_prompt_path: Optional[str] = None,
        unseen_gen_up_dir="./prompts/unseen_generalization_user_prompts/",
        num_test_up=10,
        **kwargs,
    ):
        super().__init__(
            log_dir,
            meta_train_tasks,
            meta_test_tasks,
            task_setting,
            base_model_setting,
            optim_model_setting,
            search_setting,
            init_system_prompt_path,
            **kwargs,
        )

        self.unseen_gen_up_dir = unseen_gen_up_dir
        self.num_test_up = num_test_up

        self.init_system_prompt = self.get_system_prompt(init_system_prompt_path)

        self.base_model_name = base_model_setting["model_name"]

        self.method = search_setting["method"]

    def meta_test(self):
        if self.method == "unseen_generalization":
            self.unseen_generalization()
        else:
            self.optim_method.train()

    def unseen_generalization(self):
        for task in self.task_manager.meta_test_tasks:

            ape_prompts = self.get_ape_prompts(task)
            total_nodes = []
            for instruction in ape_prompts:
                node = Node(
                    system_prompt=self.init_system_prompt,
                    instruction=instruction,
                    task=task,
                    parent=None,
                )
                self.optim_method.evaluate_node(node, split="test")
                total_nodes.append(node)
            node_data = [node.to_dict() for node in total_nodes]

            test_metric_averaged = sum(node.test_metric for node in total_nodes) / len(total_nodes)

            result_data = {
                "task_name": task.task_name,
                "system_prompt": self.init_system_prompt,
                "test_metric_averaged": test_metric_averaged,
                "node_test_metrics": [node.test_metric for node in total_nodes],
                "node_data": node_data,
            }

            self.save_data(self.log_dir, result_data, filename=f"unseen_generalization_{task.task_name}")

    def get_ape_prompts(self, task):
        file_path = f"{self.unseen_gen_up_dir}/ape_prompts_{task.task_name}.json"
        if not os.path.exists(file_path):
            self.generate_ape_prompts(task, self.num_test_up)
        try:
            with open(file_path, "r") as json_file:
                ape_prompts_data = json.load(json_file)
                return ape_prompts_data[: self.num_test_up]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

    def generate_ape_prompts(self, task):
        ape_prompts = []

        for i in range(self.num_test_up):
            demo_string = self.optim_method._get_example_ape(task, model_responses_num=10)
            new_prompt = self.optim_model.instruction_ape_generation_agent(demo=demo_string)
            new_prompt += task.suffix_prompt

            ape_prompts.append(new_prompt)

        self.save_data(
            self.unseen_gen_up_dir,
            ape_prompts,
            filename=f"ape_prompts_{task.task_name}",
        )

    def save_data(self, dir, data, filename):
        filepath = f"{dir}/{filename}.json"
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)
        self.logger.info(f"{filepath} saved")
