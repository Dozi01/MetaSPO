from .base_task import BaseDataset, BaseTask
import re
import string
from torch.utils.data import DataLoader
import numpy as np
import collections


class Grounding(BaseTask):
    def __init__(
        self,
        train_size,
        eval_size,
        test_size,
        task_name: str,
        benchmark="grounding",
        task_description="grounding tasks",
        data_dir="",
        seed=None,
        TaskDataset=BaseDataset,
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
            eval_size=eval_size,
            test_size=test_size,
            TaskDataset=TaskDataset,
            benchmark=benchmark,
            **kwargs,
        )
        self.task_name = task_name

    def clean_response(self, response):
        return response

    # To handle grounding task's answer type : list
    def _grounding_coll_func(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answers"] for item in batch]

        return {"question": questions, "answer": answers}

    def build_dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._grounding_coll_func,
        )

    def cal_correct(self, preds, labels, metric="em"):
        """
        For grounding tasks, answers are list of entities.
        """
        if not isinstance(preds, list):
            labels = [labels]
            preds = [preds]
        assert len(labels) == len(preds)

        if metric == "em":
            compute = compute_exact
        elif metric == "contain":
            compute = compute_contain

        corrects = []

        for label, pred_answer in zip(labels, preds):
            if isinstance(label, str):
                gold_entities = [label]
            elif isinstance(label, list):
                gold_entities = label
            else:
                TypeError, f"label must be str or list in Grounding tasks. Label : {label}"
            # fmt: off
            is_correct = ( 1 if np.count_nonzero([compute(gold_entity, pred_answer) for gold_entity in gold_entities]) != 0 else 0 )
            # fmt: on
            corrects.append(is_correct)

        return corrects


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_contain(gold_entity, pred_answer):
    return normalize_answer(gold_entity) in normalize_answer(pred_answer)


def f1(answers, pred_answers):
    if not isinstance(pred_answers, list):
        answers = [answers]
        pred_answers = [pred_answers]

    assert len(answers) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0
    for answer, pred_answer in zip(answers, pred_answers):
        gold_answers = set(answer)

        if len(gold_answers) == 0:
            continue

        num_all_answers += 1
        num_correct_answers += max(
            [compute_f1(gold_answer, pred_answer) for gold_answer in gold_answers]
        )

    return num_correct_answers / (num_all_answers + 1e-16)
