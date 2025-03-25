import os
import subprocess
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Device:", device)

def get_responses(prompt, tokenizer, model):
    print(prompt)
    message = (
        f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n1: "
    )
    # call llama.cpp and recover predicted string
    response = subprocess.check_output(
        f'llama.cpp/build/bin/llama-cli -m {model} -co -sp -p "{message}" -fa -ngl 80 -n 2048 --no-warmup --temp 1 -no-cnv',
        shell=True
    ).decode("utf-8").split("<|im_start|>assistant\n")[1].split("<|im_end|> [end of text]")[0]

    responses = response.split("\n")
    responses = [response for response in responses if len(response.strip()) > 0]

    return responses

# def get_responses(prompt, tokenizer, model):
#     print(prompt)
#     messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": prompt},
#         {"role": "assistant", "content": "\n1: "}
#     ]

#     model_inputs = tokenizer([
#         tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         continue_final_message=True)
#     ], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=1024
#     )

#     response = "1: " + tokenizer.batch_decode([
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ], skip_special_tokens=True)[0]

#     responses = response.split("\n")
#     responses = [response for response in responses if len(response.strip()) > 0]

#     return responses


def filter_responses(responses, answer, subject, n, prompt, tokenizer, model, filtered_responses, subject_in_response=True):
    print(responses)
    for i, response in enumerate(responses):
        if (
            response.split(":")[0] == str(i+1) and
            response.endswith(f" {answer}.") and
            (subject in response) == subject_in_response
        ):
            filtered_response = "".join(response.split(":")[1:])[:-len(f" {answer}.")].strip()
            if filtered_response not in filtered_responses:
                filtered_responses.append(filtered_response)
                if len(filtered_responses) == n:
                    return filtered_responses
    responses = get_responses(prompt, tokenizer, model)
    return filter_responses(responses, answer, subject, n, prompt, tokenizer, model, filtered_responses, subject_in_response)
    


def main():
    dataset_path = os.path.join("data", "counterfact.json")
    dataset_aug_path = os.path.join("data", "counterfact_aug.json")
    os.system(f"cp {dataset_path} {dataset_aug_path}")

    model = "./gguf_ggml_models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
    # model = "./gguf_ggml_models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    # model = "./gguf_ggml_models/DeepSeek-R1-Distill-Qwen-7B.Q4_K_M.gguf"

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(dataset_aug_path, "r") as f:
        data = json.load(f)

    for i, sample in enumerate(data):
        print(f"Processing sample {i+1}/{len(data)}")
        requested_rewrite = sample["requested_rewrite"]
        target = requested_rewrite["target_new"]["str"]
        true = requested_rewrite["target_true"]["str"]
        subject = requested_rewrite["subject"]
        edit = requested_rewrite["prompt"].format(requested_rewrite["subject"])

        # prompt_paraphrase = (
        #     f"Generate 5 distinct paraphrases of the original sentence.\n"
        #     f"The paraphrases should have a different structure and wording, but all should convey the same meaning as the original sentence.\n"
        #     f"The paraphrases should end with '{target}.'.\n"
        #     f"The paraphrases should be written in English.\n"
        #     f"Original sentence: {edit} {target}."
        # )
        # prompt_paraphrase = f"Generate 5 paraphrases that MUST each ends by ' {target}.' of the following sentence: {edit} {target}."
        prompt_paraphrase = (
            f"Generate 25 paraphrases written in English that MUST each ends by ' {target}.' "
            f"of the following sentence: {edit} {target}."
        )
        paraphrases = get_responses(prompt_paraphrase, tokenizer, model)
        filtered_paraphrases = filter_responses(paraphrases, target, subject, 50, prompt_paraphrase, tokenizer, model, [], subject_in_response=True)
        print(filtered_paraphrases)

        # prompt_neighborhood = (
        #     f"Generate 15 distinct variations of the original sentence.\n"
        #     f"In the variations, the object should be the same as the original sentence: '{true}'.\n"
        #     f"However, the subject should not be the same or a rewording of the one in the original sentence: '{subject}'.\n"
        #     f"The variations should never refer to '{subject}'.\n"
        #     f"The variations should end with '{true}.'.\n"
        #     f"Each sentence should be a true fact.\n"
        #     f"Each sentence should be written in English.\n"
        #     f"Original sentence: {edit} {true}."
        # )
        # prompt_neighborhood = f"Generate 15 similar sentences that MUST each ends by ' {true}.' where the subject '{subject}' is not the same as the original sentence but the object '{true}' is the same as the original sentence. The sentence should be a true fact: {edit} {true}."
        prompt_neighborhood = (
            f"Generate 25 similar sentences written in English that must each ends by ' {true}.'. "
            f"The subject of each generated sentence must be different from the original sentence one (i.e '{subject}'). "
            f"However, the object must be the same as the original sentence (i.e '{true}'). "
            f"All the generated sentences should be true facts with valid subjects that exist. "
            f"The relation between the subject and the object expressed in the generated sentences should also be true. "
            f"The generated sentences should have a semantic similar to the original sentence, only paraphrasing and changing the subject. "
            f"Original sentence: {edit} {true}."
        )
        neighborhoods = get_responses(prompt_neighborhood, tokenizer, model)
        filtered_neighborhoods = filter_responses(neighborhoods, true, subject, 50, prompt_neighborhood, tokenizer, model, [], subject_in_response=False)
        print(filtered_neighborhoods)

        data[i]["paraphrase_prompts_aug"] = filtered_paraphrases
        data[i]["neighborhood_prompts_aug"] = filtered_neighborhoods
        with open(dataset_aug_path, "w") as f:
            print("Writing to file")
            json.dump(data, f, indent=2)
        if i==2:
            break


if __name__ == '__main__':
    main()