# pip install peft transformers bitsandbytes
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct", quantization_config=quant_config, device_map="auto")

lora_config = LoraConfig(
    r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform=list(range(18,23)),  # зона зеркала
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, lora_config)
