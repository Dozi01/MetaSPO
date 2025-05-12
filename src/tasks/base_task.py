# define task prompts for various datasets
import os
from torch.utils.data import DataLoader
import json
import random
import numpy as np
from abc import ABC, abstractmethod

TEST_SET_SHUFFLE_SEED = 42


class BaseTask(ABC):
    def __init__(
        self,
        train_size,
        test_size,
        task_name="base_task",
        data_dir="./dataset",
        seed=None,
        benchmark=None,
        batch_size=500,
        **kwargs,
    ):
        self.task_name = task_name
        self.benchmark = benchmark

        self.data_dir = data_dir
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size

        self._get_dataset()

        self.initial_prompt, self.initial_prompt_wo_suffix, self.suffix_prompt = self._get_task_initial_prompt()

        print(f"task : {self.task_name}")
        print(f"train_size set : {len(self.train_dataset)}")
        print(f"test_size set : {len(self.test_dataset)}")

    def _get_dataset(self):
        raw_data = self._load_task_dataset()

        dataset = self._shuffle_and_split_dataset(dataset=raw_data)

        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

        self.train_data, self.test_data = self._get_data(dataset=dataset)

    def _load_task_dataset(self):
        data_file = f"{self.data_dir}/{self.benchmark}/{self.task_name}.json"

        if not (os.path.exists(data_file)):
            raise ValueError(f"json file {data_file} does not exist.")

        with open(data_file, "r") as file:
            data = json.load(file)

        return data

    def _shuffle_and_split_dataset(self, dataset, base_shuffle=True):
        assert "train" in dataset and "test" in dataset, "Dataset must contain 'train' and 'test' keys."

        train_set, test_set = dataset["train"], dataset["test"]
        assert self.train_size <= len(train_set), "train_size exceeds available training data."

        random.seed(TEST_SET_SHUFFLE_SEED)
        random.shuffle(test_set)

        if base_shuffle and self.seed is not None:
            random.seed(self.seed)
            random.shuffle(train_set)

        return dict(train=train_set[: self.train_size], test=test_set[: self.test_size])

    def _get_data(self, dataset):
        self.train_dataloader = DataLoader(dataset["train"], batch_size=self.batch_size, shuffle=False)
        train_data = next(iter(self.train_dataloader))

        self.test_dataloader = DataLoader(dataset["test"], batch_size=self.batch_size, shuffle=False)
        test_data = next(iter(self.test_dataloader))

        return train_data, test_data

    def cal_correct(self, preds, labels):
        return list(np.array((np.array(preds) == np.array(labels))).astype(int))

    def cal_metric(self, preds, labels):
        correct = self.cal_correct(preds=preds, labels=labels)
        return np.mean(correct)

    @abstractmethod
    def clean_response(self, response):
        '''
        Clean the response from the model.
        '''
        pass

    def batch_clean_responses(self, responses):
        if not isinstance(responses, list):
            responses = list(responses)

        batch_answers = [self.clean_response(response) for response in responses]
        return batch_answers

    @abstractmethod
    def _get_task_initial_prompt(self):
        '''
        Get the initial prompt for the task.
        '''
        pass
