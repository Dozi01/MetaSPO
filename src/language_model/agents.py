from .meta_prompts import *
import re
import numpy as np
from .openai_model import OpenAIModel
from .vllm_model import VLLM
from ..tasks import BaseTask


LANGUAGE_MODELS = {"openai": OpenAIModel, "vllm": VLLM}


def get_language_model(language_model_name):
    assert language_model_name in LANGUAGE_MODELS.keys(), f"Language model type {language_model_name} is not supported."
    return LANGUAGE_MODELS[language_model_name]


class BaseModel:
    def __init__(self, base_model_setting: dict, logger):

        self.base_model = get_language_model(base_model_setting["model_type"])(**base_model_setting)
        self._batch_forward_func = self.base_model.batch_forward_func
        self.logger = logger

    def forward(self, batch, cur_prompt, task: BaseTask):
        batch_size = len(batch["question"])
        # Building prompts for the forward pass and the gradient calculation
        batch_prompts = self._build_forward_prompts_completion(
            batch["question"],
            user_prompt=cur_prompt["user"],
            system_prompt=cur_prompt["system"],
        )
        for_model_input_example = self._build_prompt_for_gradient(batch_prompts)

        # Obtaining model responses and processing predictions and labels
        responses = self._batch_forward_func(batch_prompts)
        preds = task.batch_clean_responses(responses)
        labels = batch["answer"]

        # Calculating evaluation metrics
        correct = task.cal_correct(preds=preds, labels=labels)
        accuracy = np.mean(correct)
        metric = task.cal_metric(preds=preds, labels=labels)

        # Preparing examples for output
        examples = [
            {
                "cur_prompt": cur_prompt,
                "question": question,
                "answer": answer,
                "model_input": model_input,
                "model_response": response,
                "label": label,
                "pred": pred,
            }
            for question, answer, model_input, response, label, pred in zip(
                batch["question"],
                batch["answer"],
                for_model_input_example,
                responses,
                labels,
                preds,
            )
        ]

        # Constructing forward output
        forward_output = {
            "cur_prompt": cur_prompt,
            "correct": correct,
            "examples": examples,
            "acc": accuracy,
            "metric": metric,
        }

        # Logging information
        # self._log_debug_info(
        #     batch_prompts, batch["question"], responses, preds, labels, correct
        # )

        log_str = self._get_forward_log_template().format(
            task_name=task.task_name,
            cur_prompt=cur_prompt,
            num_examples=batch_size,
            metric=metric,
        )
        self.logger.info(log_str)

        return forward_output

        # Helper function for debugging information

    def _get_forward_log_template(self):
        forward_log_template = """---------------\tModel Output\t----------------\ntask_name: {task_name}\ncur_prompt:\n{cur_prompt}\nnum_examples: {num_examples}\nmetric: {metric}\n"""
        return forward_log_template

    def _log_debug_info(self, prompts, questions, responses, predictions, labels, correct, num_debug=10):
        for prom, ques, resp, pred, label, corr in zip(
            prompts[:num_debug],
            questions[:num_debug],
            responses[:num_debug],
            predictions[:num_debug],
            labels[:num_debug],
            correct[:num_debug],
        ):
            self.logger.info(
                f"Question: {ques}\nResponses :\n{resp}\nPrediction: {pred} Label: {label} Correct: {corr}\n-----\n"
            )

    def _build_forward_prompts_completion(self, questions, user_prompt, system_prompt=""):
        prompts = []

        for i, question in enumerate(questions):
            message = []
            if system_prompt != "":
                message.append({"role": "system", "content": system_prompt})
            message.append(
                {
                    "role": "user",
                    "content": f"{user_prompt.replace('{question}', question)}",
                }
            )
            prompts.append(message)

        return prompts

    def _build_prompt_for_gradient(self, batch_prompts):
        if len(batch_prompts[0]) == 1:  # no system prompt
            return [{"system": "", "user": prompt[0]["content"]} for prompt in batch_prompts]
        else:
            return [{"system": prompt[0]["content"], "user": prompt[1]["content"]} for prompt in batch_prompts]

    def _split_wrong_and_correct_examples(self, forward_output):
        wrong_examples = []
        correct_examples = []

        for i, example in enumerate(forward_output["examples"]):
            if forward_output["correct"][i] == 0:
                wrong_examples.append(example)

            elif forward_output["correct"][i] == 1:
                correct_examples.append(example)

            else:
                raise ValueError(f"_get_wrong_examples: invalid correct number {i} {forward_output}.")

        return wrong_examples, correct_examples

    def get_model_response(self, batch, cur_prompt, task):
        forward_output = self.forward(batch=batch, cur_prompt=cur_prompt, task=task)

        wrong_examples, correct_examples = self._split_wrong_and_correct_examples(forward_output=forward_output)

        return (
            wrong_examples,
            correct_examples,
            forward_output["metric"],
            forward_output,
        )


class OptimizationModel:
    def __init__(
        self,
        optim_model_setting,
        logger=None,
    ):
        self.optim_model = get_language_model(optim_model_setting["model_type"])(**optim_model_setting)
        self.logger = logger

    def log_information(self, phase: str, prompt: str, response: str) -> None:
        self.logger.info(f"--------- \t Optimize {phase} Prompt\t ---------")
        total_prompt = ""
        for role_content in prompt:
            total_prompt += f'{role_content["role"]}\n{role_content["content"]}\n'

        self.logger.info(f"{total_prompt}\n{'-' * 80}\n{response}")

    def _clean_response(self, optim_response, tag_name):
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        matches = re.findall(pattern=pattern, string=optim_response, flags=re.DOTALL)
        for i, m in enumerate(matches):
            matches[i] = m.strip()
        if not matches:
            return "N/A : Format wrong"  # response format wrong

        if isinstance(matches, list):
            return matches[0]
        else:
            return matches

    def _generate_analysis(self, system_propmt, template: str, **kwargs) -> str:
        analysis_prompt = template.format(**kwargs)
        prompt = self._build_prompt(system_propmt, analysis_prompt)
        response = self.optim_model.generate(prompt)
        analysis = self._clean_response(response, tag_name="Analysis")
        return analysis

    def _optimize_instruction(
        self,
        system_propmt: str,
        template: str,
        tag_name="improved_instruction_prompt",
        **kwargs,
    ) -> str:
        optimize_prompt = template.format(**kwargs)
        prompt = self._build_prompt(system_propmt, optimize_prompt)
        response = self.optim_model.generate(prompt)
        improved_instruction = self._clean_response(response, tag_name=tag_name)
        self.log_information("Instruction", prompt, response)
        return improved_instruction

    def instruction_ape_generation_agent(self, demo):
        instruction = self._optimize_instruction(
            None, template=ape_generation_template, demo=demo, tag_name="instruction"
        )
        return instruction

    def instruction_ape_resampling_agent(self, instruction):
        return self._optimize_instruction(
            None,
            template=ape_resampling_template,
            instruction=instruction,
            question="{question}",
        )

    def instruction_writer_agent(self, system_prompt: str, instruction: str, examples_string: str) -> str:
        analysis = self._generate_analysis(
            gradient_instruction_writer_system_prompt,
            gradient_for_instruction_writer_template,
            system_prompt=system_prompt,
            instruction=instruction,
            examples=examples_string,
        )

        return self._optimize_instruction(
            optimizer_instruction_writer_system_prompt,
            optimizer_instruction_writer_template,
            system_prompt=system_prompt,
            instruction=instruction,
            examples=examples_string,
            analysis=analysis,
            question="{question}",
        )

    def _optimize_system_prompt(self, system_propmt, template: str, **kwargs) -> str:
        optimize_prompt = template.format(**kwargs)
        prompt = self._build_prompt(system_propmt, optimize_prompt)
        response = self.optim_model.generate(prompt)
        updated_system_prompt = self._clean_response(response, tag_name="improved_system_prompt")
        self.log_information("System", prompt, response)
        return updated_system_prompt

    def system_writer_agent(self, current_system_prompt: str, example_strings: list) -> str:
        analysis = self._generate_analysis(
            gradient_system_writer_system_prompt,
            gradient_for_system_writer_template,
            system_prompt=current_system_prompt,
            examples=example_strings,
        )
        return self._optimize_system_prompt(
            optimizer_system_writer_system_prompt,
            optimizer_system_writer_template,
            system_prompt=current_system_prompt,
            analysis=analysis,
        )

    def _build_prompt(self, system_prompt, user_prompt):
        if system_prompt == None:
            return [{"role": "user", "content": user_prompt}]

        else:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return prompt
