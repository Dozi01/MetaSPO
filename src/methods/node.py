import itertools
import random
from typing import Optional
from ..tasks import BaseTask


class Node:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        system_prompt: str,
        instruction: str,
        task: BaseTask = None,
        parent: Optional["Node"] = None,
        test_metric: float = -1,
        eval_metric: float = -1,
    ):

        self.id = next(Node.id_iter)
        self.system_prompt = system_prompt
        self.instruction = instruction

        self.parent = parent
        self.test_metric = test_metric
        self.eval_metric = eval_metric

        if parent is None:
            self.depth = 0
            assert task is not None
            self.task = task
        else:
            self.depth = parent.depth + 1
            self.task = parent.task

    def update_model_wrong_response(self, responses):
        self.model_wrong_responses = []
        self.model_wrong_responses.extend(responses)

    def update_model_correct_response(self, responses):
        self.model_correct_responses = []
        self.model_correct_responses.extend(responses)

    def _sample_wrong_examples(self, model_responses_num):
        num_wrong_examples = len(self.model_wrong_responses)
        if num_wrong_examples < model_responses_num:
            sampled_examples = self.model_wrong_responses
        else:
            sampled_examples = random.sample(self.model_wrong_responses, model_responses_num)

        return sampled_examples

    def _format_answer(self, example):
        return ", ".join(example["label"]) if isinstance(example["label"], list) else example["label"]

    def get_example_string(self, model_responses_num: int = 3):
        examples = self._sample_wrong_examples(model_responses_num)
        example_strings = []

        for example in examples:
            system_prompt = example["model_input"]["system"]

            example_string = self._get_example_template().format(
                system_prompt=system_prompt,
                user_prompt=example["model_input"]["user"],
                label=self._format_answer(example),
                response=example["model_response"],
                prediction=example["pred"],
            )
            example_strings.append(example_string)

        return "\n".join(example_strings)

    def _get_example_template(self):
        example_template = """<Example>\nSystem prompt: \n{system_prompt}\n\nUser prompt:\n{user_prompt}\n\nResponse: \n{response}\n\nPrediction: \n{prediction}\n\nThe correct label is : \n{label}\n</Example>""".strip()
        return example_template

    def to_dict(self):
        """
        Converts the node object to a dictionary format for logging.

        Returns:
            dict: A dictionary containing the node's attributes.
        """
        return {
            "id": self.id,
            "task": self.task.task_name,
            "system_prompt": self.system_prompt,
            "instruction": self.instruction,
            "parent_id": self.parent.id if self.parent else None,
            "depth": self.depth,
            "eval_metric": self.eval_metric,
            "test_metric": self.test_metric,
        }
