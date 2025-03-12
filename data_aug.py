import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_response(prompt, tokenizer, model):
    print(prompt)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "\n1: "}
    ]

    model_inputs = tokenizer([
        tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True)
    ], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    response = "1: " + tokenizer.batch_decode([
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ], skip_special_tokens=True)[0]

    responses = response.split("\n")
    responses = [response for response in responses if len(response.strip()) > 0]

    return responses


def filter_responses(responses, answer, subject, n, subject_in_response=True):
    print(responses)
    filtered_responses = []
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
                    break
    if len(filtered_responses) != n:
        raise Exception(f"Failed to generate {n} sentences", responses)
    print(filtered_responses)
    return filtered_responses


def main():
    dataset_path = os.path.join("data", "counterfact.json")
    dataset_aug_path = os.path.join("data", "counterfact_aug.json")
    os.system(f"cp {dataset_path} {dataset_aug_path}")

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
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

            prompt_paraphrase = f"Generate 10 paraphrases that MUST each ends by ' {target}.' of the following sentence: {edit} {target}."
            paraphrases = get_response(prompt_paraphrase, tokenizer, model)
            filtered_paraphrases = filter_responses(paraphrases, target, subject, 2, subject_in_response=True)

            prompt_neighborhood = f"Generate 15 similar sentences that MUST each ends by ' {true}.' where the subject '{subject}' is not the same as the original sentence but the object '{true}' is the same as the original sentence. The sentence should be a true fact: {edit} {true}."
            neighborhoods = get_response(prompt_neighborhood, tokenizer, model)
            filtered_neighborhoods = filter_responses(neighborhoods, true, subject, 10, subject_in_response=False)

            sample["paraphrase_prompts_aug"] = filtered_paraphrases
            sample["neighborhood_prompts_aug"] = filtered_neighborhoods
    with open(dataset_aug_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()