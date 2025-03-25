from copy import deepcopy
from typing import Dict, List, Tuple
import subprocess
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .modeledit_hparams import ModelEditHyperParams

import hfedit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_modeledit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ModelEditHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for _, request in enumerate(requests):
        model, tok = execute_modeledit(model, tok, request, hparams)
        print(f"New weights successfully inserted")

    return model, weights_copy

def execute_modeledit(model, tok, request, hparams):
    requested_rewrite = request["requested_rewrite"]

    target = requested_rewrite["target_new"]["str"]
    true = requested_rewrite["target_true"]["str"]

    request["paraphrase_prompts_aug"] = request["paraphrase_prompts_aug"][:10]
    request["neighborhood_prompts_aug"] = request["neighborhood_prompts_aug"][:10]

    prompt = requested_rewrite["prompt"].format(requested_rewrite["subject"])
    if hparams.n_tok_start == -1 or hparams.n_tok_start == -3:
        n_tok_prompt = [tok(f" {prompt}", return_length=True)["length"][0]]
    else:
        prompt_after_subject = prompt.split(requested_rewrite["subject"])[-1]
        n_tok_prompt = [tok(prompt_after_subject, return_length=True)["length"][0] + 1]

    if hparams.method == "multi" or hparams.method == "generalisation":
        # instruction = "Complete the following{0} sentence"
        instruction = "Complete in a single sentence."
        # gld_prompt = [f"{instruction.format(' 2')}s. {prompt} {target}. {prompt}"]
        gld_prompt = [f"{instruction} {prompt} {target}. {prompt}"]
        gld_length = tok(gld_prompt[-1], return_length=True)["length"][0]
        # err_lenght = tok(f"{instruction.format('')}. {prompt}", return_length=True)["length"][0]
        # err_prompt = [f"{instruction.format('')}. {(gld_length - err_lenght)*'_ '}{prompt}"]
        err_lenght = tok(f"{instruction} {prompt}", return_length=True)["length"][0]
        err_prompt = [f"{instruction} {(gld_length - err_lenght)*'_ '}{prompt}"]

        for p in request["paraphrase_prompts_aug"]:
            # gld_prompt.append(f"{instruction.format(' 2')}s. {p} {target}. {p}")
            gld_prompt.append(f"{instruction} {p} {target}. {p}")
            gld_length = tok(gld_prompt[-1], return_length=True)["length"][0]
            # err_lenght = tok(f"{instruction.format('')}. {p}", return_length=True)["length"][0]
            # err_prompt.append(f"{instruction.format('')}. {(gld_length - err_lenght)*'_ '}{p}")
            err_lenght = tok(f"{instruction} {p}", return_length=True)["length"][0]
            err_prompt.append(f"{instruction} {(gld_length - err_lenght)*'_ '}{p}")
            if hparams.n_tok_start == -1 or hparams.n_tok_start == -3:
                n_tok_prompt.append(tok(f" {p}", return_length=True)["length"][0])
            else:
                prompt_after_subject = p.split(requested_rewrite["subject"])[-1]
                n_tok_prompt.append(tok(prompt_after_subject, return_length=True)["length"][0] + 1)

        # We don't have access to subject
        if hparams.method == "multi":
            for p in request["neighborhood_prompts_aug"]:
                # gld_prompt.append(f"{instruction.format(' 2')}s. {p} {true}. {p}")
                gld_prompt.append(f"{instruction} {p} {true}. {p}")
                gld_length = tok(gld_prompt[-1], return_length=True)["length"][0]
                # err_lenght = tok(f"{instruction.format('')}. {p}", return_length=True)["length"][0]
                # err_prompt.append(f"{instruction.format('')}. {(gld_length - err_lenght)*'_ '}{p}")
                err_lenght = tok(f"{instruction} {p}", return_length=True)["length"][0]
                err_prompt.append(f"{instruction} {(gld_length - err_lenght)*'_ '}{p}")
                n_tok_prompt.append(tok(f" {p}", return_length=True)["length"][0])

    else:
        if hparams.method == "icl":
            # instruction = "Complete the following{0} prompt"
            instruction = "Complete in a single sentence."
            n_paraphrase = len(request["paraphrase_prompts_aug"])
            n_neighborhood = len(request["neighborhood_prompts_aug"])
            # paraphrase_prompts = "".join([f"New fact: {prompt} {target}.\nPrompt {i+1}: {p} {target}.\n" for i, p in enumerate(request["paraphrase_prompts"])])
            paraphrase_prompts = "".join([f"{p} {target}. " for i, p in enumerate(request["paraphrase_prompts_aug"])])
            # neighborhood_prompts = "".join([f"New fact: {prompt} {target}.\nPrompt {i+n_paraphrase+1}: {p} {true}.\n" for i, p in enumerate(request["neighborhood_prompts"])])
            neighborhood_prompts = "".join([f"{p} {true}. " for i, p in enumerate(request["neighborhood_prompts_aug"])])
            # gld_prompt = [f"{instruction.format(' ' + str(n_paraphrase+n_neighborhood+1))}s.\n{paraphrase_prompts}{neighborhood_prompts}New fact: {prompt} {target}.\nPrompt {n_paraphrase+n_neighborhood+1}: {prompt}"]
            # gld_prompt = [f"{instruction}\n{paraphrase_prompts}{neighborhood_prompts}New fact: {prompt} {target}.\nPrompt {n_paraphrase+n_neighborhood+1}: {prompt}"]
            gld_prompt = [f"{instruction} {paraphrase_prompts} {neighborhood_prompts} {prompt} {target}. {prompt}"]
            gld_prompt = [f"{instruction} {paraphrase_prompts} {prompt} {target}. {prompt}"]
        elif hparams.method == "classic":
            # instruction = "Complete the following{0} sentence"
            # gld_prompt = [f"{instruction.format(' 2')}s. {prompt} {target}. {prompt}"]
            # instruction = "Complete the following sentence"
            instruction = "Complete in a single sentence."
            gld_prompt = [f"{instruction} {prompt} {target}. {prompt}"]
        else:
            raise Exception("Method must be multi, icl, generalisation or classic")

        gld_length = tok(gld_prompt[-1], return_length=True)["length"][0]
        # err_lenght = tok(f"{instruction.format('')}. {prompt}", return_length=True)["length"][0]
        # err_prompt = [f"{instruction.format('')}. {(gld_length - err_lenght)*'_ '}{prompt}"]
        err_lenght = tok(f"{instruction} {prompt}", return_length=True)["length"][0]
        err_prompt = [f"{instruction} {(gld_length - err_lenght)*'_ '}{prompt}"]
        print(gld_length, tok(err_prompt[-1], return_length=True)["length"][0])
        
    model, tokenizer = hfedit.main(deepcopy(model), tok, gld_prompt, err_prompt, n_tok_prompt, hparams.n_tok_start, hparams.n_tok_stop, hparams.insertion_type, hparams.layer_to_modify, hparams.strength, hparams.treshold)
    
    return model, {}