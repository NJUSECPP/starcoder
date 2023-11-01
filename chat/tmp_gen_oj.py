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
import requests
from dialogues import DialogueTemplate, get_dialogue_template
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

code_result_status_map = {
    -1: "COMPILE_FAILED",
    0: "SUCCESS",
    1: "CPU_TIME_LIMIT_EXCEEDED",
    2: "REAL_TIME_LIMIT_EXCEEDED",
    3: "MEMORY_LIMIT_EXCEEDED",
    4: "RUNTIME_ERROR",
    5: "SYSTEM_ERROR"
}

class OJTester:
    def __init__(self, filepath="question.txt"):
        self.session = requests.Session()
        self.base_url = "http://172.29.4.19:8082"
        self.filepath = filepath
        self.login()

    def login(self):
        login_url = f"{self.base_url}/web/login"
        login_data = {
            "username": "gpt_test_user",
            "password": "123456"
        }
        response = self.session.post(login_url, data=login_data)
        if response.status_code != 200 or "error" in response.json():
            raise Exception("Failed to login to OJ")

    def get_expected_input_output(self, question):
        with open(self.filepath, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                sliced_line = line.strip()[: len(question)]
                if sliced_line == question:
                    return lines[i+1].strip(), lines[i+2].strip()
        return None, None

    def run_and_check_code(self, code, question):
        expected_input, expected_output = self.get_expected_input_output(question)
        if not expected_input or not expected_output:
            return "Question not found in the file.", -1000

        OJ_API_URL = f"{self.base_url}/test/self"
        payload = {
            "examId": 10032,
            "questionId": 10079,
            "username": "gpt_test_user",
            "code": "#include <bits/stdc++.h>\n"+code,
            "input": expected_input
        }

        try:
            response = self.session.post(OJ_API_URL, json=payload)
            if self.is_session_expired(response):
                self.login()
                response = self.session.post(OJ_API_URL, json=payload)
            response_data = response.json()

            if response_data["state"] == "OK":
                object_data = eval(response_data["object"])
                code_result_status = object_data['code_result_status']
                if code_result_status == -1:
                    flag = -1
                    error_message = object_data['compileErrDetail']
                    message = (
                        f"Your code syntax is incorrect. The code cannot be compiled successfully, "
                        f"and the compiler reports an error that:\n{error_message}"
                    )
                elif code_result_status == 0:
                    if object_data['output'].strip() == expected_output:
                        flag = 1
                        message = "Congratulations, your code is correct and can pass the test case!"
                    else:
                        flag = -1
                        message = "Your code is wrong and cannot pass the test case."
                else:
                    flag = -1
                    error_message = code_result_status_map[code_result_status]
                    message = (
                        f"Your code is incorrect. The code does not run correctly, "
                        f"and it reports an error that:\n{error_message}"
                    )
            else:
                message = f"Error from OJ: {response_data['result']}"
                flag = -1

        except Exception as e:
            message = f"An error occurred during code execution: {str(e)}"
            flag = -1

        return message, flag

    def is_session_expired(self, response):
        return response.status_code == 401


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
        "--system_prompt",
        type=str,
        default=None,
        help="Overrides the dialogue template's system prompt",
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    try:
        dialogue_template = DialogueTemplate.from_pretrained(
            args.model_id, revision=args.revision
        )
    except Exception:
        print(
            "No dialogue template found in model repo. Defaulting to the `no_system` template."
        )
        dialogue_template = get_dialogue_template("no_system")

    if args.system_prompt is not None:
        dialogue_template.system = args.system_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(
        f"EOS token ID for generation: {tokenizer.convert_tokens_to_ids(dialogue_template.end_token)}"
    )
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
        args.model_id,
        revision=args.revision,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    flag = 1  # 0: failed; 1: success
    dialogue_template.messages = []
    oj_tester = OJTester()
    question = ""

    while True:
        if flag == 1:
            print("User: ", end="")
            user_input = input()
            if user_input == "exit":
                break
            else:
                question = user_input.split("Requirements:", 1)[0].strip()
                dialogue_template.messages.append(
                    {"role": "user", "content": user_input}
                )

        prompt = dialogue_template.get_inference_prompt()
        batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
            device
        )
        generated_ids = model.generate(**batch, generation_config=generation_config)
        generated_text = (
            tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            .lstrip()
            .split("<|assistant|>\n")[-1]
        )
        dialogue_template.messages.append(
            {"role": "assistant", "content": generated_text}
        )
        print()
        print("Assistant: ", generated_text)
        print()
        print("======================")
        print()

        # 提取代码部分
        code_start = generated_text.find("```")
        code_end = generated_text.find("```", code_start + 3)
        running_result = ""
        if code_start != -1 and code_end != -1:
            while generated_text[code_start] != "\n":
                code_start += 1
            code = generated_text[code_start + 1 : code_end].strip()  # 提取代码内容
            running_result, flag = oj_tester.run_and_check_code(code, question)
            print("User: " + running_result)
            dialogue_template.messages.append(
                {"role": "user", "content": running_result}
            )
        
        if running_result == "" and flag == -1:
            print("User: Please give me the new code. Note a few possible errors: you can't include extra output.")
            dialogue_template.messages.append(
                {"role": "user", "content": "Please give me the new code. Note a few possible errors: you can't include extra output."}
            )


        # 如果问题在txt中不存在，则退出
        if flag == -1000:
            break

        if flag == 1:
            prompt = dialogue_template.get_inference_prompt()
            batch = tokenizer(
                prompt, return_tensors="pt", return_token_type_ids=False
            ).to(device)
            generated_ids = model.generate(**batch, generation_config=generation_config)
            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=False
            ).lstrip()
            generated_text = generated_text.split("<|assistant|>\n")[-1]

            print("Assistant: ", generated_text)

            # 提取助手的回应内容，并存储在assistant_output中
            dialogue_template.messages.append(
                {"role": "assistant", "content": generated_text}
            )


if __name__ == "__main__":
    main()
