# define task prompts for various datasets
from .base_task import BaseDataset, BaseTask
import re
import string


class MEDMCQA(BaseTask):
    def __init__(
        self,
        train_size,
        eval_size,
        test_size,
        task_name: str,
        benchmark="medmcqa",
        task_description="medical question answering tasks",
        data_dir="",
        seed=None,
        TaskDataset=BaseDataset,
        option_num=5,
        **kwargs,
    ):
        self.options = {}
        super().__init__(
            task_name=task_name,
            task_description=task_description,
            data_dir=data_dir,
            seed=seed,
            train_size=train_size,
            eval_size=eval_size,
            test_size=test_size,
            benchmark=benchmark,
            TaskDataset=TaskDataset,
            option_num=option_num,
            **kwargs,
        )

    def clean_response(self, response):
        valid_options = (
            string.ascii_uppercase[: self.option_num]
            + string.ascii_lowercase[: self.option_num]
        )
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
