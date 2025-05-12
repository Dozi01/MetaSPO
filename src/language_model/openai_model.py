from openai import OpenAI
import time

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
        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.gpt_chat_completion

    def batch_forward_chatcompletion(self, batch_prompts):
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
