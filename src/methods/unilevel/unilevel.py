import json
from ...language_model import BaseModel, OptimizationModel
from ...taskmanager import TaskManager
from ..node import Node
import os


class Unilevel:
    def __init__(
        self,
        initial_system_prompt,
        task_manager: TaskManager,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        log_dir,
        method: str,
        logger,
        iteration=3,
        model_responses_num: int = 3,
        **kwargs,
    ) -> None:

        self.task_manager = task_manager
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model
        self.initial_system_prompt = initial_system_prompt

        self.model_responses_num = model_responses_num
        self.method = method
        self.iteration = iteration
        self.log_dir = log_dir

        self.train_log = list()
        self.test_log = list()

    def train(self):
        raise NotImplementedError

    def optimize_system_only(self, node: Node):
        example_strings = node.get_example_string()

        updated_system_prompt = self.optim_model.system_writer_agent(node.system_prompt, example_strings)
        new_node = Node(
            system_prompt=updated_system_prompt,
            instruction=node.instruction,
            parent=node,
        )
        return new_node

    def optimize_user_only(self, node: Node):

        examples_string = node.get_example_string()

        system_prompt = node.system_prompt

        updated_instruction = self.optim_model.instruction_writer_agent(
            system_prompt=system_prompt,
            instruction=node.instruction,
            examples_string=examples_string,
        )

        new_node = Node(
            system_prompt=node.system_prompt,
            instruction=updated_instruction,
            parent=node,
        )
        return new_node

    def evaluate_node(self, node: Node, split):
        metric, model_wrong_examples, model_correct_examples = self.evaluate_prompt(
            system=node.system_prompt,
            user=node.instruction,
            task=node.task,
            split=split,
        )

        if split == "train":
            node.eval_metric = metric
            node.update_model_correct_response(model_correct_examples)
            node.update_model_wrong_response(model_wrong_examples)
        if split == "test":
            node.test_metric = metric

    def evaluate_prompt(self, system, user, task, split="train"):
        if split not in ["train", "test"]:
            raise ValueError("Invalid split specified. Use 'train' or 'test'.")

        # Select data based on the split
        data = task.train_data if split == "train" else task.test_data

        # Construct prompt
        current_prompt = {
            "system": system,
            "user": user,
        }

        # Get model response and evaluation
        wrong_examples, correct_examples, metric, forward_output = self.base_model.get_model_response(
            data, current_prompt, task=task
        )

        return metric, wrong_examples, correct_examples

    def write_log(self, system_prompt, user_prompt, task, metric, split):
        log = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "split": split,
            "task": task.task_name,
            "score": metric,
        }

        self._append_log(split, log)

    def _append_log(self, split, log):
        """
        Appends the log entry to the correct log based on split type.
        """
        if split == "train":
            self.train_log.append(log)
        elif split == "test":
            self.test_log.append(log)

    def save_log(self):
        """
        Saves the training and test logs to JSON files.
        """
        self._save_to_file("train_log.json", self.train_log)
        self._save_to_file("test_log.json", self.test_log)

    def _save_to_file(self, filename, data):
        """
        Writes log data to a specified JSON file.
        """
        file_path = f"{self.log_dir}/{filename}"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def save_data(self, nodes_data, filename: str = "nodes"):
        log_dir_parts = self.log_dir.split(os.sep)
        if len(log_dir_parts) > 1:
            log_dir_parts.pop()
        new_log_dir = os.sep.join(log_dir_parts)

        version = 0
        base_filename = f"unilevel_{filename}_{version}.json"
        full_path = os.path.join(new_log_dir, base_filename)

        while os.path.exists(full_path):
            version += 1
            base_filename = f"unilevel_{filename}_{version}.json"
            full_path = os.path.join(new_log_dir, base_filename)

        with open(full_path, "w") as file:
            json.dump(nodes_data, file, indent=4)

        self.logger.info(f"Save log: {full_path}")
