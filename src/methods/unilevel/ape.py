from ...language_model import BaseModel, OptimizationModel
from ...taskmanager import TaskManager
from ...tasks import BaseTask
from .unilevel import Unilevel
import random


class APE(Unilevel):
    def __init__(
        self,
        initial_system_prompt,
        task_manager: TaskManager,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        log_dir,
        method: str,
        logger,
        iteration,
        top_k,
        print_log: bool = True,
        model_responses_num: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
            initial_system_prompt,
            task_manager,
            base_model,
            optim_model,
            log_dir,
            method,
            logger,
            iteration=iteration,
            top_k=top_k,
            print_log=print_log,
            model_responses_num=model_responses_num,
        )

        self.top_k = top_k

    def train(self):
        assert self.method == "ape", "APE class method must be 'ape'"
        for task in self.task_manager.meta_test_tasks:
            self.optimize_ape(task)

    def optimize_ape(self, task: BaseTask):

        self.initialize_prompt(task)

        self.test_prompt(task)

        # Initial Proposal Step
        candidate_prompts = [(task.initial_prompt, task.best_metric)]
        batch_candidates = [(task.initial_prompt, task.best_metric)]

        for _ in range(self.iteration):
            demo_string = self._get_example_ape(task, model_responses_num=10)
            new_prompt = self.optim_model.instruction_ape_generation_agent(demo=demo_string)

            new_prompt += task.suffix_prompt

            metric, _, _ = self.evaluate_prompt(system=task.system_prompt, user=new_prompt, task=task, split="train")
            batch_candidates.append((new_prompt, metric))

        candidate_prompts.extend(batch_candidates)
        candidate_prompts.sort(key=lambda x: x[1], reverse=True)
        top_prompts = candidate_prompts[: self.top_k]

        # Iterative Proposal Step
        for prompt, _ in top_prompts:
            batch_candidates = []

            for _ in range(self.iteration):

                # Generate a new prompt based on the current top prompt
                new_prompt = self.optim_model.instruction_ape_resampling_agent(prompt)

                # Evaluate the newly generated prompt
                metric, _, _ = self.evaluate_prompt(
                    system=task.system_prompt,
                    user=new_prompt,
                    task=task,
                    split="train",
                )
                batch_candidates.append((new_prompt, metric))

            candidate_prompts.extend(batch_candidates)

        # Sort all candidate prompts generated during the iterative proposal step by scores
        candidate_prompts.sort(key=lambda x: x[1], reverse=True)

        # Select the top prompt based on evaluation scores as the optimized prompt
        best_prompt, best_metric = candidate_prompts[0]

        task.current_prompt = best_prompt

        self.test_prompt(task)

        self.save_log()

    def initialize_prompt(self, task: BaseTask):
        task.system_prompt = self.initial_system_prompt
        task.current_prompt = task.initial_prompt
        metric, model_wrong_examples, model_correct_examples = self.evaluate_prompt(
            system=task.system_prompt,
            user=task.current_prompt,
            task=task,
            split="train",
        )
        task.best_metric = metric
        self.write_log(
            system_prompt=task.system_prompt,
            user_prompt=task.current_prompt,
            task=task,
            metric=metric,
            split="train",
        )

    def test_prompt(self, task):
        metric, model_wrong_examples, model_correct_examples = self.evaluate_prompt(
            system=task.system_prompt,
            user=task.current_prompt,
            task=task,
            split="test",
        )

        self.write_log(
            system_prompt=task.system_prompt,
            user_prompt=task.current_prompt,
            task=task,
            metric=metric,
            split="test",
        )

        return metric

    def _format_answer(self, example):
        return ", ".join(example["answer"]) if isinstance(example["answer"], list) else example["answer"]

    def _get_example_ape(self, task: BaseTask, model_responses_num=10):
        questions = task.train_data["question"]
        answers = task.train_data["answer"]

        # Ensuring we do not exceed the available number of questions/answers
        num_examples = min(len(questions), model_responses_num)

        indices = random.sample(range(len(questions)), num_examples)

        example_strings = [
            self._qa_example_template(
                question=questions[i],
                answer=self._format_answer({"answer": answers[i]}),
            )
            for i in indices
        ]

        return "\n".join(example_strings)

    def _qa_example_template(self, question, answer):
        return f"Input :\n{question}\nOutput :\n{answer}\n"
