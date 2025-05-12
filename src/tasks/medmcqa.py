# define task prompts for various datasets
from .base_task import BaseTask
import re
import string


class MEDMCQA(BaseTask):
    def __init__(
        self,
        train_size,
        test_size,
        task_name: str,
        benchmark="medmcqa",
        task_description="medical question answering tasks",
        data_dir="",
        seed=None,
        option_num=4,
        **kwargs,
    ):
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

        self.option_num = option_num

    def _get_task_initial_prompt(self):
        base_prompt = "Given the following question and candidate answers, choose the best answer."
        suffix = "<Question>{question}</Question>\nAt the end present your answer in <answer> and </answer>."
        initial_prompt = base_prompt + suffix
        return initial_prompt, base_prompt, suffix

    def clean_response(self, response):
        valid_options = string.ascii_uppercase[: self.option_num] + string.ascii_lowercase[: self.option_num]
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"

        matches = re.findall(clean_pattern, response, re.IGNORECASE)

        if not matches:
            return "N/A: Format error"

        last_match = matches[-1]
        answer = re.search(r"\(([{}]?)\)".format(valid_options), last_match)
        if answer:
            return answer.group(1).upper()

        answer = re.search(r"[{}]".format(valid_options), last_match)
        if answer:
            return answer.group(0).upper()

        return "N/A: Format error"
