from ...language_model import BaseModel, OptimizationModel
from ...taskmanager import TaskManager
from ...tasks import BaseTask
from ..node import Node
import random
from .metaspo import BilevelNodes, MetaSPO


class MetaSPOAPE(MetaSPO):
    def __init__(
        self,
        initial_system_prompt,
        task_manager: TaskManager,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        log_dir,
        logger,
        method: str,
        system_num_candidate=18,
        user_num_candidate=3,
        iteration=3,
        **kwargs,
    ) -> None:
        super().__init__(
            initial_system_prompt=initial_system_prompt,
            task_manager=task_manager,
            base_model=base_model,
            optim_model=optim_model,
            log_dir=log_dir,
            logger=logger,
            method=method,
            iteration=iteration,
            system_num_candidate=system_num_candidate,
            user_num_candidate=user_num_candidate,
            **kwargs,
        )

        self.task_manager = task_manager
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model
        self.initial_system_prompt = initial_system_prompt
        self.system_prompt = initial_system_prompt

        self.method = method
        self.all_greater = False
        self.beam_cut = True
        self.iteration = iteration
        self.log_dir = log_dir

        self.train_log = list()
        self.test_log = list()

    def train(self):
        self.method_metaspo_ape()

    def method_metaspo_ape(self):
        self.run_meta_training(
            optimize_system_fn=self.optimize_system_ape,
            optimize_user_fn=self.optimize_user_ape,
        )

    def optimize_user_ape(self, node: Node):
        demo_string = self._generate_demo_string([node.task], model_responses_num=10)
        new_prompt = self.optim_model.instruction_ape_generation_agent(demo=demo_string)
        new_prompt += node.task.suffix_prompt

        new_node = Node(
            system_prompt=node.system_prompt,
            instruction=new_prompt,
            parent=node,
        )
        return new_node

    def optimize_system_ape(self, bilevel_nodes: BilevelNodes):
        tasks = [nodes[0].task for nodes in bilevel_nodes.node_list]
        demo_string = self._generate_demo_string(tasks, model_responses_num=5)

        new_system_prompt = self.optim_model.instruction_ape_generation_agent(demo=demo_string)

        new_node_list = [
            [Node(new_system_prompt, node.instruction, parent=node) for node in nodes]
            for nodes in bilevel_nodes.node_list
        ]

        # Initialize the new BilevelNodes with the updated node list
        new_bilevel_nodes = BilevelNodes(new_node_list, bilevel_nodes.beam_width)

        return new_bilevel_nodes

    def _generate_demo_string(self, task_list, model_responses_num):
        example_string_list = []
        for task in task_list:
            example_string_list += self._get_example_ape(task, model_responses_num)
        random.shuffle(example_string_list)
        return "\n".join(example_string_list)

    def _format_answer(self, example):
        return ", ".join(example["answer"]) if isinstance(example["answer"], list) else example["answer"]

    def _get_example_ape(self, task: BaseTask, model_responses_num=10):
        questions = task.train_data["question"]
        answers = task.train_data["answer"]

        # Ensuring we do not exceed the available number of questions/answers
        num_examples = min(len(questions), model_responses_num)

        indices = random.sample(range(len(questions)), num_examples)

        example_strings = [
            self._qa_example_template(
                question=questions[i],
                answer=self._format_answer({"answer": answers[i]}),
            )
            for i in indices
        ]

        return example_strings

    def _qa_example_template(self, question, answer):
        return f"Input :\n{question}\nOutput :\n{answer}\n"
