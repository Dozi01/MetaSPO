import json
from .methods import *
from .runner import Runner
from .methods.node import Node
import os


class Analyser(Runner):
    def __init__(
        self,
        args,
    ):
        super().__init__(args)

        self.unseen_gen_up_dir = args.unseen_gen_up_dir
        self.num_test_up = args.num_test_up
        self.analysis_method = args.analysis_method

    def meta_test(self):
        if self.analysis_method == "unseen_generalization":
            self.unseen_generalization()
        elif self.analysis_method == "test_time_adaptation":
            # This will call ProTeGi with optimized system prompt
            self.optim_method.train()

    def unseen_generalization(self):
        for task in self.task_manager.meta_test_tasks:

            user_prompts = self.get_user_prompts(task)
            total_nodes = []
            for instruction in user_prompts:
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

            self.save_data(self.log_dir, result_data, filename=f"result_unseen_gen_{task.task_name}")

    def get_user_prompts(self, task):
        file_path = f"{self.unseen_gen_up_dir}/up_for_unseen_gen_{task.task_name}.json"
        if not os.path.exists(file_path):
            self.generate_user_prompts(task)
        try:
            with open(file_path, "r") as json_file:
                user_prompts_data = json.load(json_file)
                return user_prompts_data[: self.num_test_up]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

    def generate_user_prompts(self, task):
        user_prompts = []

        for i in range(self.num_test_up):
            demo_string = self.optim_method._get_example_ape(task, model_responses_num=10)
            new_prompt = self.optim_model.instruction_ape_generation_agent(demo=demo_string)
            new_prompt += task.suffix_prompt

            user_prompts.append(new_prompt)

        self.save_data(
            self.unseen_gen_up_dir,
            user_prompts,
            filename=f"up_for_unseen_gen_{task.task_name}",
        )

    def save_data(self, dir, data, filename):
        filepath = f"{dir}/{filename}.json"
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)
        self.logger.info(f"{filepath} saved")
