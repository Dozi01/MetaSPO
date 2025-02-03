from transformers import AutoTokenizer
import torch
from typing import List
from vllm import LLM, SamplingParams

MODEL_DICT = {
    "llama3.1_8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2_3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen2.5_7B": "Qwen/Qwen2.5-7B-Instruct",
}


class VLLM:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_model_len: int = 3000,
        dtype: str = "float16",
        num_gpus: int = torch.cuda.device_count(),
        gpu_memory_utilization: float = 0.90,
        **kwargs,
    ):
        self.model_name = self._get_model_name(model_name)
        self.temperature = temperature
        self.params = self._create_sampling_params(max_model_len)
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model(max_model_len, dtype, num_gpus, gpu_memory_utilization)

    def _get_model_name(self, model_name: str) -> str:
        if model_name in MODEL_DICT:
            return MODEL_DICT[model_name]
        raise ValueError(f"Model {model_name} not supported.")

    def _create_sampling_params(self, max_model_len: int) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=max_model_len,
            skip_special_tokens=False,
            detokenize=True,
        )

    def _initialize_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, truncate=True, padding=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _initialize_model(
        self,
        max_model_len: int,
        dtype: str,
        num_gpus: int,
        gpu_memory_utilization: float,
    ) -> LLM:
        return LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def batch_forward_func(self, batch_prompts, use_tqdm=True) -> List[str]:
        batch_prompts = self.prepare_batch_prompts(batch_prompts)
        request_outputs = self.model.generate(batch_prompts, self.params, use_tqdm=use_tqdm)
        return self.postprocess_output(request_outputs)

    def generate(self, prompt: str) -> str:
        input = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        request_outputs = self.model.generate([input], self.params)
        return self.postprocess_output(request_outputs)[0]

    def postprocess_output(self, request_outputs) -> List[str]:
        return [output.outputs[0].text for output in request_outputs]

    def prepare_batch_prompts(self, batch_prompts) -> List[str]:
        return [
            self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in batch_prompts
        ]
