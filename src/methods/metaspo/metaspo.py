import json

from ...language_model import BaseModel, OptimizationModel
from ...taskmanager import TaskManager
from ..node import Node
from typing import List, Optional
import os
import random


class BilevelNodes:
    def __init__(self, node_list: Optional[List[List[Node]]], user_top_k: int):
        self.node_list = node_list
        self.user_top_k = user_top_k
        self.system_prompt = node_list[0][0].system_prompt

    @property
    def total_nodes(self):
        all_nodes = []
        for sublist in self.node_list:
            for node in sublist:
                all_nodes.append(node)
        return all_nodes

    @property
    def export_node_list(self):
        return [sublist[:] for sublist in self.node_list]

    @property
    def meta_score(self):
        total_score = 0
        total_count = 0
        for nodes in self.node_list:
            total_score += sum(node.eval_metric for node in nodes)
            total_count += len(nodes)

        return total_score / total_count if total_count > 0 else 0

    def sort_nodes(self):
        for nodes in self.node_list:
            nodes.sort(key=lambda x: x.eval_metric, reverse=True)

    def cut_nodes_with_beam(self):
        for i, nodes in enumerate(self.node_list):
            self.node_list[i] = nodes[: self.user_top_k]

    def build_example_strings_for_meta_train_set(self, example_num_per_task: int = 2):
        examples = [nodes[0].get_example_string(example_num_per_task) for nodes in self.node_list]
        random.shuffle(examples)  # Shuffle the list
        return "\n".join(examples)


class MetaSPO:
    def __init__(
        self,
        initial_system_prompt,
        task_manager: TaskManager,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        log_dir,
        logger,
        method: str,
        iteration,
        num_system_candidate,
        num_user_candidate,
        user_top_k=3,
        **kwargs,
    ) -> None:

        self.task_manager = task_manager
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model
        self.initial_system_prompt = initial_system_prompt
        self.system_prompt = initial_system_prompt

        self.method = method
        self.all_greater = False
        self.iteration = iteration
        self.num_system_candidate = num_system_candidate
        self.num_user_candidate = num_user_candidate

        self.user_top_k = user_top_k
        self.log_dir = log_dir

        self.train_log = list()
        self.test_log = list()

    def train(self):
        if self.method == "metaspo":
            self.method_metaspo()
        elif self.method == "outer_loop":
            self.method_outer_loop()

    def method_outer_loop(self):
        self.iteration = self.iteration * 2
        self.run_meta_training(optimize_system_fn=self.optimize_system, optimize_user_fn=None)

    def method_metaspo(self):
        self.run_meta_training(
            optimize_system_fn=self.optimize_system,
            optimize_user_fn=self.optimize_user,
        )

    def run_meta_training(
        self,
        optimize_system_fn=None,
        optimize_user_fn=None,
    ):
        bilevel_nodes = self.init_bilevel_nodes()
        for node in bilevel_nodes.total_nodes:
            self.evaluate_node(node=node, split="train")

        updated_nodes = [bilevel_nodes.export_node_list]

        # Optimization loop
        for iter in range(self.iteration):
            bilevel_nodes = self.inner_loop(optimize_user_fn, bilevel_nodes, updated_nodes)
            bilevel_nodes = self.outer_loop(optimize_system_fn, bilevel_nodes, updated_nodes)

        # Test only the last nodes.
        last_bilevel_node = updated_nodes[-1]
        for nodes in last_bilevel_node:
            for node in nodes:
                if node.test_metric == -1:
                    self.evaluate_node(node=node, split="test")

        def process_node(node):
            return node.to_dict()

        updated_nodes_data = [
            [[(process_node(node)) for node in node_list] for node_list in bilevel_nodes]
            for bilevel_nodes in updated_nodes
        ]

        self.system_prompt = bilevel_nodes.system_prompt

        total_data = {
            "optimized_system_prompt": self.system_prompt,
            "nodes": updated_nodes_data,
        }

        self.logger.info(f"======= OPTIMIZED SYSTEM PROMPT =======")
        self.logger.info(self.system_prompt)

        self.save_data(total_data)

    def inner_loop(self, optimize_user_fn, bilevel_nodes, updated_nodes):
        if not optimize_user_fn:
            return bilevel_nodes
        
        self.logger.info(f"======= INNER LOOP =======")

        for nodes in bilevel_nodes.node_list:
            for _ in range(self.num_user_candidate):
                new_node = optimize_user_fn(nodes[0])  # we can use multiple nodes in this step.
                self.evaluate_node(node=new_node, split="train")
                nodes.append(new_node)

        bilevel_nodes.sort_nodes()
        bilevel_nodes.cut_nodes_with_beam()

        updated_nodes.append(bilevel_nodes.export_node_list)

        return bilevel_nodes

    def outer_loop(self, optimize_system_fn, bilevel_nodes, updated_nodes):
        if not optimize_system_fn:
            return bilevel_nodes

        self.logger.info(f"======= OUTER LOOP =======")

        new_bilevel_nodes_list = [optimize_system_fn(bilevel_nodes) for _ in range(self.num_system_candidate)]

        for new_bilevel_nodes in new_bilevel_nodes_list:
            for node in new_bilevel_nodes.total_nodes:
                self.evaluate_node(node=node, split="train")

            if new_bilevel_nodes.meta_score > bilevel_nodes.meta_score:
                new_bilevel_nodes.sort_nodes()
                bilevel_nodes = new_bilevel_nodes

        updated_nodes.append(bilevel_nodes.export_node_list)

        return bilevel_nodes

    def optimize_user(self, node: Node):

        examples_string = node.get_example_string()

        updated_instruction = self.optim_model.instruction_writer_agent(
            system_prompt=node.system_prompt,
            instruction=node.instruction,
            examples_string=examples_string,
        )

        new_node = Node(
            system_prompt=node.system_prompt,
            instruction=updated_instruction,
            parent=node,
        )
        return new_node

    def optimize_system(self, bilevel_nodes: BilevelNodes):
        example_strings = bilevel_nodes.build_example_strings_for_meta_train_set()

        updated_system_prompt = self.optim_model.system_writer_agent(bilevel_nodes.system_prompt, example_strings)

        new_node_list = [
            [Node(updated_system_prompt, node.instruction, parent=node) for node in nodes]
            for nodes in bilevel_nodes.node_list
        ]

        # Initialize the new BilevelNodes with the updated node list
        new_bilevel_nodes = BilevelNodes(new_node_list, bilevel_nodes.user_top_k)

        return new_bilevel_nodes

    def init_bilevel_nodes(self):
        return BilevelNodes(
            [
                [
                    Node(
                        self.initial_system_prompt,
                        task.initial_prompt,
                        task=task,
                        parent=None,
                    )
                ]
                for task in self.task_manager.tasks
            ],
            user_top_k=self.user_top_k,
        )

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

    def evaluate_node(self, node: Node, split):
        metric, model_wrong_examples, model_correct_examples = self.evaluate_prompt(
            system=node.system_prompt,
            user=node.instruction,
            task=node.task,
            split=split,
        )

        if split == "train":
            node.eval_metric = metric
            # node.update_model_correct_response(model_correct_examples)
            node.update_model_wrong_response(model_wrong_examples)
        if split == "test":
            node.test_metric = metric

    def save_data(self, nodes_data, filename: str = "nodes"):
        log_dir_parts = self.log_dir.split(os.sep)
        if len(log_dir_parts) > 1:
            log_dir_parts.pop()
        new_log_dir = os.sep.join(log_dir_parts)

        version = 0
        base_filename = f"bilevel_{filename}_{version}.json"
        full_path = os.path.join(new_log_dir, base_filename)

        while os.path.exists(full_path):
            version += 1
            base_filename = f"bilevel_{filename}_{version}.json"
            full_path = os.path.join(new_log_dir, base_filename)

        with open(full_path, "w") as file:
            json.dump(nodes_data, file, indent=4)

        self.logger.info(f"Save log: {full_path}")
