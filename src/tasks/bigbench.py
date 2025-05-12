# define task prompts for various datasets
from .base_task import BaseTask
import re
import string
import json
import os


TASKS_ANSWER_IS_OPTION = [
    "logical_deduction",
    "temporal_sequences",
    "tracking_shuffled_objects",
]


INITIAL_PROMPTS = {
    "logic_grid_puzzle": {
        "base_prompt": "Solve logic grid puzzles",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer bracketed between <answer> and </answer>.",
    },
    "logical_deduction": {
        "base_prompt": "A logical deduction task which requires deducing the order of a sequence of objects.",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer option bracketed between <answer> and </answer>.",
    },
    "object_counting": {
        "base_prompt": "Questions that involve enumerating objects and asking the model to count them.",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer bracketed between <answer> and </answer>.",
    },
    "reasoning_colored_objects": {
        "base_prompt": "Answer extremely simple questions about the colors of objects on a surface.",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer bracketed between <answer> and </answer>.",
    },
    "temporal_sequences": {
        "base_prompt": "Answer questions about which times certain events could have occurred.",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer option bracketed between <answer> and </answer>.",
    },
    "tracking_shuffled_objects": {
        "base_prompt": "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer option bracketed between <answer> and </answer>.",
    },
    "epistemic": {
        "base_prompt": "Determine whether one sentence entails the next.",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer option bracketed between <answer> and </answer>.",
    },
    "navigate": {
        "base_prompt": "Given a series of navigation instructions, determine whether one would end up back at the starting point.",
        "suffix": "<Question>{question}</Question>\nAt the end show the answer option bracketed between <answer> and </answer>.",
    },
}


class Bigbench(BaseTask):
    def __init__(
        self,
        train_size,
        test_size,
        task_name: str,
        benchmark="bigbench",
        task_description="task from bigbench",
        data_dir="",
        seed=None,
        option_num=7,
        **kwargs,
    ):
        self.benchmark = benchmark
        super().__init__(
            task_name=task_name,
            task_description=task_description,
            data_dir=data_dir,
            seed=seed,
            train_size=train_size,
            test_size=test_size,
            option_num=option_num,
            benchmark=benchmark,
            **kwargs,
        )

        self.option_num = option_num

        self.task_name = task_name

        self.number_to_word_dict = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "twenty-one": 21,
        }

    def load_task_dataset(self):

        data_file = f"{self.data_dir}/{self.benchmark}/{self.task_name}.json"

        if not (os.path.exists(data_file)):
            raise ValueError(f"json file {data_file} does not exist.")

        with open(data_file, "r") as file:
            data = json.load(file)

        return data

    def _get_task_initial_prompt(self):
        base_prompt = INITIAL_PROMPTS[self.task_name]["base_prompt"]
        suffix = INITIAL_PROMPTS[self.task_name]["suffix"]
        initial_prompt = base_prompt + suffix
        return initial_prompt, base_prompt, suffix

    def clean_response(self, response):
        if self.task_name in TASKS_ANSWER_IS_OPTION:
            return self.clean_response_options(response)
        else:
            return self.clean_response_non_option(response)

    def clean_response_options(self, response):
        letters = string.ascii_lowercase[: self.option_num]
        # Regex pattern to extract content within <answer> tags
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"

        # Find all matches and get the last non-empty match
        matches = re.findall(clean_pattern, response.lower())
        if not matches or not matches[-1].strip():
            return "N/A: Format error"

        answer_content = matches[-1].strip().lower()

        # Attempt to find patterns of type (X) or standalone letters
        patterns = [r"\(([" + letters + r"])\)", r"[" + letters + r"]"]

        for pattern in patterns:
            match = re.search(pattern, answer_content)
            if match:
                return match.group(0).strip("()").upper()

        # If no valid pattern is found, return a format error
        return "N/A: Format error"

    def clean_response_non_option(self, response):
        # Regex pattern to extract content within <answer> tags
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"

        # Find all matches and get the last non-empty match
        matches = re.findall(clean_pattern, response.lower())
        if not matches or not matches[-1].strip():
            return "N/A: Format error"

        answer_content = matches[-1].strip().lower()

        if answer_content in self.number_to_word_dict:
            return self.number_to_word_dict[answer_content]

        return answer_content
