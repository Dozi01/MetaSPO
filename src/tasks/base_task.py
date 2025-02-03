# define task prompts for various datasets
import re
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import random
import numpy as np

import string
from abc import ABC, abstractmethod

TEST_SET_SHUFFLE_SEED = 42


class BaseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class BaseTask(ABC):
    def __init__(
        self,
        train_size,
        eval_size,
        test_size,
        task_name="base_task",
        data_dir="./dataset",
        seed=None,
        TaskDataset=BaseDataset,
        benchmark=None,
        option_num=5,
        initial_user_prompt: str = None,
        batch_size=500,
        prompt_file_path="./prompts/task_initial_prompts.json", # path to initial prompts
        **kwargs,
    ):
        self.task_name = task_name
        self.data_dir = data_dir
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.eval_size = eval_size
        self.batch_size = batch_size
        self.TaskDataset = TaskDataset
        self.option_num = option_num
        self.benchmark = benchmark
        self.prompt_file_path = prompt_file_path
        self.initial_user_prompt = initial_user_prompt

        self.dataset = self.get_split_task_dataset(
            dataset=self.load_task_dataset(),
            seed=seed,
            train_size=train_size,
            eval_size=eval_size,
            test_size=test_size,
            base_shuffle=True,
        )
        self.train_dataset = self.dataset["train"]
        self.eval_dataset = self.dataset["eval"]
        self.test_dataset = self.dataset["test"]
        self._get_task_dataset()
        self.initial_prompt, self.initial_prompt_wo_suffix, self.suffix_prompt = (
            self._get_task_initial_prompt()
        )

        print(f"task : {self.task_name}")
        print(f"train_size set: {len(self.train_dataset)}")
        print(f"test_size set: {len(self.test_dataset)}")

        self.answer_format_prompt = None

    def load_task_dataset(self):

        data_file = f"{self.data_dir}/{self.benchmark}/{self.task_name}.json"

        if not (os.path.exists(data_file)):
            raise ValueError(f"json file {data_file} does not exist.")

        with open(data_file, "r") as file:
            data = json.load(file)

        return data

    def cal_correct(self, preds, labels):
        return list(np.array((np.array(preds) == np.array(labels))).astype(int))

    def cal_metric(self, preds, labels):
        """
        <task specific>
        """
        correct = self.cal_correct(preds=preds, labels=labels)
        return np.mean(correct)

    def clean_labels(self, labels):
        return labels

    @abstractmethod
    def clean_response(self, response):
        pass

    def batch_clean_responses(self, responses):
        """
        Extract preds from responses.
        """
        if not isinstance(responses, list):
            responses = list(responses)
        batch_answers = []
        for response in responses:
            batch_answers.append(self.clean_response(response))
        return batch_answers

    def get_split_task_dataset(
        self,
        dataset,
        train_size,
        eval_size,
        test_size,
        seed=None,
        base_shuffle=True,
    ):
        """
        Split the dataset into training set, eval set and testing set.
        """

        assert isinstance(dataset, dict), "Dataset must be a dictionary."
        assert (
            "train" in dataset and "test" in dataset
        ), "Dataset must contain 'train' and 'test' keys."

        train_set = dataset["train"]
        test_set = dataset["test"]

        total_train_size = len(train_set)
        assert train_size + eval_size <= total_train_size, (
            f"train_size ({train_size}) + eval_size ({eval_size}) "
            f"exceeds the total available training data ({total_train_size})."
        )

        print(f"Shuffling dataset TEST set. Seed: {TEST_SET_SHUFFLE_SEED}")
        random.seed(TEST_SET_SHUFFLE_SEED)
        random.shuffle(test_set)

        if base_shuffle and seed is not None:
            print(f"Shuffling dataset for train/validation set. Seed: {seed}")
            random.seed(seed)
            random.shuffle(train_set)

        eval_set = train_set[-eval_size:]
        train_set = train_set[:train_size]
        test_set = test_set[:test_size]

        dataset = dict(train=train_set, eval=eval_set, test=test_set)
        return dataset

    def build_task_dataset(self, dataset, TaskDataset=None):
        return TaskDataset(dataset=dataset)

    def build_dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_dataloader(self, split, batch_size, shuffle=False):
        if self.TaskDataset is None:
            self.TaskDataset = BaseDataset

        if split not in self.dataset.keys():
            raise ValueError(f"Dataset split {split} does not exist.")

        dataset = self.build_task_dataset(
            self.dataset[split], TaskDataset=self.TaskDataset
        )

        return self.build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _get_task_dataset(self):
        self.train_dataloader = self.get_dataloader(
            "train", batch_size=self.batch_size, shuffle=False
        )
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)
        self.train_data = next(self.train_data_iterator)

        self.test_dataloader = self.get_dataloader(
            "test", batch_size=self.batch_size, shuffle=False
        )
        self.test_data_iterator = self._infinite_data_loader(self.test_dataloader)
        self.test_data = next(self.test_data_iterator)

    def _infinite_data_loader(self, data_loader):
        """
        Yield batches from dataloader.
        """
        while True:
            for batch in data_loader:
                yield batch

    def _get_task_initial_prompt(self):
        if not os.path.exists(self.prompt_file_path):
            raise ValueError(f"json file {self.prompt_file_path} does not exist.")

        with open(self.prompt_file_path, "r", encoding="utf-8") as file:
            initial_prompts = json.load(file)

        # Retrieve the prompt for the specified task name
        task_initial_prompt = initial_prompts.get(self.task_name)
        if not task_initial_prompt:
            raise KeyError(
                f"Task name '{self.task_name}' not found in the initial prompt."
            )

        # Construct the key for the initial prompt
        initial_prompt_key = f"{self.initial_user_prompt}_prompt"

        # Check if the required keys are present in the task's prompt
        required_keys = [initial_prompt_key, "suffix"]
        missing_keys = [key for key in required_keys if key not in task_initial_prompt]

        if missing_keys:
            raise KeyError(f"Missing keys {missing_keys} for task '{self.task_name}'.")

        intial_prompt = (
            task_initial_prompt[initial_prompt_key] + task_initial_prompt["suffix"]
        )

        return (
            intial_prompt,
            task_initial_prompt[initial_prompt_key],
            task_initial_prompt["suffix"],
        )
