from ...language_model import BaseModel, OptimizationModel
from ...taskmanager import TaskManager
from ...tasks import BaseTask
from .unilevel import Unilevel
from ..node import Node


class APO(Unilevel):
    def __init__(
        self,
        initial_system_prompt: str,
        task_manager: TaskManager,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        log_dir,
        method: str,
        logger,
        default_system_prompt_when_optimize: bool = False,
        beam_width=3,
        iteration=6,
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

        self.beam_width = beam_width
        self.default_system_prompt_when_optimize = default_system_prompt_when_optimize

    def train(self):
        assert self.method == "apo", "APO class method must be 'apo'"
        for task in self.task_manager.meta_test_tasks:
            self.optimize_apo(task)

    def optimize_apo(self, task: BaseTask):
        node = Node(self.initial_system_prompt, task.initial_prompt, task=task, parent=None)
        self.evaluate_node(node=node, split="train")

        total_node = [node]
        updated_node = [node]

        batch_candidates = []
        # candidate_nodes should contain initial node.
        candidate_nodes = [node]
        for _ in range(self.beam_width):

            new_node = self.optimize_user_only(
                node=node,
                default_system_prompt=self.default_system_prompt_when_optimize,
            )
            self.evaluate_node(node=new_node, split="train")

            total_node.append(new_node)
            batch_candidates.append(new_node)

        candidate_nodes.extend(batch_candidates)
        candidate_nodes.sort(key=lambda x: x.eval_metric, reverse=True)
        candidate_nodes = candidate_nodes[: self.beam_width]

        updated_node.extend(candidate_nodes)

        for iter in range(self.iteration):
            batch_candidates = []

            # Each prompt in current top prompts is used to produce new candidates
            for node in candidate_nodes:
                for _ in range(self.beam_width):
                    new_node = self.optimize_user_only(
                        node=node,
                        default_system_prompt=self.default_system_prompt_when_optimize,
                    )
                    self.evaluate_node(node=new_node, split="train")

                    total_node.append(new_node)
                    batch_candidates.append(new_node)

            candidate_nodes.extend(batch_candidates)
            candidate_nodes.sort(key=lambda x: x.eval_metric, reverse=True)
            candidate_nodes = candidate_nodes[: self.beam_width]
            updated_node.extend(candidate_nodes)

        for node in updated_node:
            if node.test_metric == -1:
                self.evaluate_node(node=node, split="test")

        nodes_data = [node.to_dict() for node in updated_node]
        self.save_data(nodes_data, filename=f"{task.task_name}")

        total_nodes_data = [node.to_dict() for node in total_node]
        self.save_data(total_nodes_data, filename=f"{task.task_name}_total")

        return
