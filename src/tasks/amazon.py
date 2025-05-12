# define task prompts for various datasets
from .base_task import BaseTask
import re


class Amazon(BaseTask):
    def __init__(
        self,
        train_size,
        test_size,
        task_name: str,
        benchmark="amazon",
        task_description="Amazon review analysis benchmark",
        data_dir="",
        seed=None,
        **kwargs,
    ):
        self.options = {}
        self.benchmark = benchmark
        super().__init__(
            task_name=task_name,
            task_description=task_description,
            data_dir=data_dir,
            seed=seed,
            train_size=train_size,
            test_size=test_size,
            benchmark=benchmark,
            **kwargs,
        )

        self.task_name = task_name

    def _get_task_initial_prompt(self):
        base_prompt = "Predict the customer's rating from 1 to 5."
        suffix = "<Question>{question}</Question>\nAt the end present your answer in <answer> and </answer>."
        initial_prompt = base_prompt + suffix
        return initial_prompt, base_prompt, suffix

    def clean_response(self, response):
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"

        matches = re.findall(clean_pattern, response.lower())

        # no answer in response.
        if not matches or not matches[-1].strip():
            return -1

        digits = re.findall(r"\d+", matches[-1])

        if not digits:
            return -1
        else:
            return int(digits[0])
