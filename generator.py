"""
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. 
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, 
of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any 
copyright notation herein.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMGenerator():
    def truncate(self, text, length=128):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])

    def generate_assessments(self, prompt, class_names):
        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            relevance_judgement_logits = self.model(**model_inputs).logits
            last_token_logits = relevance_judgement_logits[:, -1, :]
            zero_token_id = self.tokenizer.convert_tokens_to_ids(class_names[0])
            one_token_id = self.tokenizer.convert_tokens_to_ids(class_names[1])
            zero_logits = last_token_logits[:, zero_token_id]
            one_logits = last_token_logits[:, one_token_id]
            probabilities = F.softmax(torch.tensor([zero_logits, one_logits]), dim=-1)
            return probabilities

    def generate_synthetic_passages(self, prompt, num_return_sequences=8):
        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=num_return_sequences)
            pseudo_passages = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = [self.clean_outputs(pseudo_passage) for pseudo_passage in pseudo_passages]
            return response    

class MistralGenerator(LLMGenerator):
    def __init__(self, model_path):
        self.device = "cuda"
        self.model_path = model_path
        if 'Mixtral' in model_path:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def clean_outputs(self, generation):
        generation = generation.split("[/INST]")[-1].strip()
        return generation

class GemmaGenerator(LLMGenerator):
    def __init__(self, model_path):
        self.device = "cuda"
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def clean_outputs(self, generation):
        generation = generation.split("model\n")[-1].strip()
        return generation

class Llama3Generator(LLMGenerator):
    def __init__(self, model_path):
        self.device = "cuda"
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate_assessments(self, prompt, class_names):
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Relevance Category:"},
                    ]
        model_inputs = self.tokenizer.apply_chat_template(messages, continue_final_message=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            relevance_judgement_logits = self.model(**model_inputs).logits
            last_token_logits = relevance_judgement_logits[:, -1, :]
            zero_token_id = self.tokenizer.convert_tokens_to_ids(class_names[0])
            one_token_id = self.tokenizer.convert_tokens_to_ids(class_names[1])
            zero_logits = last_token_logits[:, zero_token_id]
            one_logits = last_token_logits[:, one_token_id]
            probabilities = F.softmax(torch.tensor([zero_logits, one_logits]), dim=-1)
            return probabilities

    def clean_outputs(self, generation):
        generation = generation.split("assistant\n")[-1].strip()
        return generation