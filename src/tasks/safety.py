# define task prompts for various datasets
from .base_task import BaseDataset, BaseTask
import re
import string
import os
import json
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Safety(BaseTask):
    def __init__(
        self,
        train_size,
        eval_size,
        test_size,
        task_name: str,
        benchmark="safety",
        task_description="LLM Safety benchmark",
        data_dir="",
        seed=None,
        TaskDataset=BaseDataset,
        f1_metric=True,
        **kwargs,
    ):
        self.options = {}
        self.benchmark = benchmark
        self.f1_metric = f1_metric
        super().__init__(
            task_name=task_name,
            task_description=task_description,
            data_dir=data_dir,
            seed=seed,
            train_size=train_size,
            eval_size=eval_size,
            test_size=test_size,
            TaskDataset=TaskDataset,
            benchmark=benchmark,
            **kwargs,
        )

        self.task_name = task_name

    def clean_response(self, response):
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        match = re.findall(clean_pattern, response.lower())

        if len(match) == 0 or not match[-1].strip():
            return "N/A: Format error"

        return match[-1].strip().lower()

    def cal_metric(self, preds, labels):
        correct = self.cal_correct(preds=preds, labels=labels)
        accuracy = np.mean(correct)

        if self.f1_metric:
            # Calculating F1 Score, Precision, and Recall
            f1 = f1_score(labels, preds, average="macro")
            return f1

        return accuracy
