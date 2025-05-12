from ...language_model import BaseModel, OptimizationModel
from ...taskmanager import TaskManager
from ...tasks import BaseTask
from .unilevel import Unilevel
from ..node import Node


class ProTeGi(Unilevel):
    def __init__(
        self,
        initial_system_prompt: str,
        task_manager: TaskManager,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        log_dir,
        method: str,
        logger,
        iteration=6,
        num_user_candidate=3,
        print_log: bool = True,
        model_responses_num: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            initial_system_prompt,
            task_manager,
            base_model,
            optim_model,
            log_dir,
            method,
            logger,
            iteration=iteration,
            print_log=print_log,
            model_responses_num=model_responses_num,
        )

        self.num_user_candidate = num_user_candidate

    def train(self):
        assert self.method == "protegi", "ProTeGi class method must be 'protegi'"
        for task in self.task_manager.meta_test_tasks:
            self.optimize_protegi(task)

    def optimize_protegi(self, task: BaseTask):
        node = Node(self.initial_system_prompt, task.initial_prompt, task=task, parent=None)
        self.evaluate_node(node=node, split="train")

        total_node = [node]
        updated_node = [node]

        batch_candidates = []
        # candidate_nodes should contain initial node.
        candidate_nodes = [node]
        for _ in range(self.num_user_candidate):

            new_node = self.optimize_user_only(node=node)
            self.evaluate_node(node=new_node, split="train")

            total_node.append(new_node)
            batch_candidates.append(new_node)

        candidate_nodes.extend(batch_candidates)
        candidate_nodes.sort(key=lambda x: x.eval_metric, reverse=True)
        candidate_nodes = candidate_nodes[: self.num_user_candidate]

        updated_node.extend(candidate_nodes)

        for iter in range(self.iteration):
            batch_candidates = []

            for node in candidate_nodes:
                for _ in range(self.num_user_candidate):
                    new_node = self.optimize_user_only(node=node)
                    self.evaluate_node(node=new_node, split="train")

                    total_node.append(new_node)
                    batch_candidates.append(new_node)

            candidate_nodes.extend(batch_candidates)
            candidate_nodes.sort(key=lambda x: x.eval_metric, reverse=True)
            candidate_nodes = candidate_nodes[: self.num_user_candidate]
            updated_node.extend(candidate_nodes)

        for node in updated_node:
            if node.test_metric == -1:
                self.evaluate_node(node=node, split="test")

        nodes_data = [node.to_dict() for node in updated_node]
        self.save_data(nodes_data, filename=f"{task.task_name}")