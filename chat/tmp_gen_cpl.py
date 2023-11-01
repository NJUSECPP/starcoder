# coding=utf-8
# Copyright 2023 The BigCode and HuggingFace teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""A simple script to quickly check the model outputs of a generative model"""
import argparse

import torch
from dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, set_seed)

import subprocess



def compile_and_check_code(code):
    flag = 0
    code_filename = "user_code.cpp"

    with open(code_filename, "w") as code_file:
        code_file.write("#include <bits/stdc++.h>\n"+code)


    try:
        result = subprocess.run(["g++", code_filename, "-o", "compiled_code"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            message = "Your code is correct. The code can be compiled successfully."
            flag = 1
        else:
            error_message = result.stderr
            message = (f"Your code is incorrect. The code cannot be compiled successfully, "
                       f"and the compiler reports an error that:\n{error_message}")
    except Exception as e:
        message = f"An error occurred during compilation: {str(e)}"

    return message, flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="Name of model to generate samples with",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="The model repo's revision to use",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None, help="Overrides the dialogue template's system prompt"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_id, revision=args.revision)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")

    if args.system_prompt is not None:
        dialogue_template.system = args.system_prompt




    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"EOS token ID for generation: {tokenizer.convert_tokens_to_ids(dialogue_template.end_token)}")
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=4096,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )


    
    flag = 1 # 0: failed; 1: success
    dialogue_template.messages = []
    while True:

        if flag == 1:
            print("User: ", end="")
            user_input = input()
            if user_input == "exit":
                break
            else:
                dialogue_template.messages.append({"role": "user", "content": user_input})

        prompt = dialogue_template.get_inference_prompt()
        batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        generated_ids = model.generate(**batch, generation_config=generation_config)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip().split("<|assistant|>\n")[-1]
        dialogue_template.messages.append({"role": "assistant", "content": generated_text})
        print()
        print("Assistant: ", generated_text)
        print()
        print("======================")
        print()

        # 提取代码部分
        code_start = generated_text.find("```")
        code_end = generated_text.find("```", code_start + 3)
        tmp = 0
        if code_start != -1 and code_end != -1:
            tmp = 1
            while generated_text[code_start] != '\n':
                code_start += 1
            code = generated_text[code_start + 1 : code_end].strip()  # 提取代码内容
            compile_result, flag = compile_and_check_code(code)
            print("User: " + compile_result)
            dialogue_template.messages.append({"role": "user", "content": compile_result})

        if tmp == 0:
            compile_result = "Please give me the new code. Note some possible errors: use backquotes to comment your code at the beginning and end, and your program must not contain extra output."
            print("User: " + compile_result)
            dialogue_template.messages.append({"role": "user", "content": compile_result})

        if flag == 1:
            prompt = dialogue_template.get_inference_prompt()
            batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
            generated_ids = model.generate(**batch, generation_config=generation_config)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()
            generated_text = generated_text.split("<|assistant|>\n")[-1]

            print("Assistant: ", generated_text)
            
            # 提取助手的回应内容，并存储在assistant_output中
            dialogue_template.messages.append({"role": "assistant", "content": generated_text})

if __name__ == "__main__":
    main()

