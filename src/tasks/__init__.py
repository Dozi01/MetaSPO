import importlib
from .base_task import BaseTask

GROUNDING_TASKS = [
    "hotpot_qa",
    "natural_questions",
    "squad",
    "web_qa",
    "drop",
    "trivia_qa",
]
SAFETY_TASKS = [
    "ethos",
    "liar",
    "hatecheck",
    "sarcasm",
    "tweet_eval",
    "antropic_harmless",
]
MEDMCQA_TASKS = [
    "anatomy",
    "surgery",
    "ob_gyn",
    "medicine",
    "pharmacology",
    "dental",
    "pediatrics",
    "pathology",
]

AMAZON_TASKS = ["beauty", "game", "baby", "office", "sports", "electronics", "pet"]

BIGBENCH_TASKS = [
    "logic_grid_puzzle",
    "logical_deduction",
    "temporal_sequences",
    "tracking_shuffled_objects",
    "object_counting",
    "reasoning_colored_objects",
    "epistemic",
    "navigate",
]


def get_task(task_name):
    if task_name in GROUNDING_TASKS:
        class_name = "Grounding"
    elif task_name in SAFETY_TASKS:
        class_name = "Safety"
    elif task_name in BIGBENCH_TASKS:
        class_name = "Bigbench"
    elif task_name in MEDMCQA_TASKS:
        class_name = "MEDMCQA"
    elif task_name in AMAZON_TASKS:
        class_name = "Amazon"
    else:
        raise ValueError(f"{task_name} is not a recognized task")

    try:
        module = importlib.import_module(f".{class_name.lower()}", package=__package__)
        CustomTask = getattr(module, class_name)

    except ModuleNotFoundError:
        raise ValueError(f"Module for task '{task_name}' could not be found.")

    return CustomTask
