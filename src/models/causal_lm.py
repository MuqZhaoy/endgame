from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CausalLM:
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, dtype: torch.dtype) -> "CausalLM":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return cls(model, tokenizer) 