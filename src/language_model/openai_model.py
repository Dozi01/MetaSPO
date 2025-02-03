from openai import OpenAI
import time
import json, os

MODEL_DICT = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
}


class OpenAIModel:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        batch_mode: bool = True,
        batch_folder: str = "./logs/openai",
        **kwargs,
    ):
        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        try:
            self.model = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Init openai client error: \n{e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e

        if model_name not in MODEL_DICT:
            raise ValueError(f"Model {model_name} not supported.")

        self.model_name = MODEL_DICT[model_name]
        self.temperature = temperature
        self.batch_mode = batch_mode
        self.batch_folder = batch_folder
        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.gpt_chat_completion

    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts to openai chat API and retrieve the answers.
        Batch mode is activated when the batch_mode is true and batch has more than 10 prompts.
        """
        if self.batch_mode and len(batch_prompts) > 10:
            return self.batch_inference(batch_prompts)
        return [self.gpt_chat_completion(prompt=prompt) for prompt in batch_prompts]

    def gpt_chat_completion(self, prompt):
        backoff_time = 1
        while True:
            try:
                response = self.model.chat.completions.create(
                    messages=prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5

    def create_task(self, prompt, task_id):
        return {
            "custom_id": f"{task_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": prompt,
                "seed": 42,
            },
        }

    def write_tasks_to_file(self, tasks, file_name):
        with open(file_name, "w") as file:
            for task in tasks:
                file.write(json.dumps(task) + "\n")

    def create_batch_file(self, file_name):
        return self.model.files.create(file=open(file_name, "rb"), purpose="batch")

    def create_batch_job(self, batch_file):
        backoff_time = 60
        while True:
            try:
                return self.model.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
            except Exception as e:
                print(e, f"Rate Limit. Sleeping {backoff_time} seconds...")
                time.sleep(backoff_time)

    def monitor_batch_job(self, batch_job):
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(5)
            print(
                f"Batch job status: {batch_job.status}...trying again in 5 seconds..."
            )
            batch_job = self.model.batches.retrieve(batch_job.id)
        return batch_job

    def download_results(self, batch_job):
        result_file_id = batch_job.output_file_id
        result = self.model.files.content(result_file_id).content
        result_file_name = f"./logs/openai/batch_job_results_{batch_job.id}.jsonl"
        with open(result_file_name, "wb") as file:
            file.write(result)
        return result_file_name

    def load_results(self, result_file_name):
        results = []
        with open(result_file_name, "r") as file:
            for line in file:
                json_object = json.loads(line.strip())
                results.append(json_object)
        return results

    def batch_inference(self, batch_prompts):
        tasks = [
            self.create_task(prompt, task_id)
            for task_id, prompt in enumerate(batch_prompts)
        ]

        if not os.path.exists(self.batch_folder):
            os.makedirs(self.batch_folder)

        inf_num = len(os.listdir(self.batch_folder))
        file_name = f"{self.batch_folder}/batch_inference_{inf_num}.jsonl"
        self.write_tasks_to_file(tasks, file_name)

        batch_file = self.create_batch_file(file_name)
        batch_job = self.create_batch_job(batch_file)
        batch_job = self.monitor_batch_job(batch_job)

        if batch_job.status == "completed":
            result_file_name = self.download_results(batch_job)
            results = self.load_results(result_file_name)
            sorted_results = sorted(results, key=lambda x: int(x["custom_id"]))
            return [
                response["response"]["body"]["choices"][0]["message"]["content"]
                for response in sorted_results
            ]
        else:
            raise OpenaiBatchError


class OpenaiBatchError(Exception):
    def __str__(self):
        return "Openai Batch API Inference Error"
