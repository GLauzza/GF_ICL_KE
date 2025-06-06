import struct
import json
import sys
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from modified_models.modified_qwen2 import Qwen2ModifiedForCausalLM, Qwen2ModifiedConfig, Qwen2MLP
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def extract_activations(model, tokenizer, prompts, prompts_labels, activations, n_tok_prompt, n_tok_start, n_tok_stop):
    def hook_inp(model, input, output):
        temp_activations["inp"].append(input[0].detach())
    def hook_out(model, input, output):
        temp_activations["out"].append(output.detach())
    def hook_residual(model, input, output):
        temp_activations["residual"].append(input[0].detach())

    handles = []
    for n, m in model.named_modules():
        if n.endswith(".mlp.gate_proj"):
            handles.append(m.register_forward_hook(hook_inp))
        elif n.endswith(".mlp.down_proj"):
            handles.append(m.register_forward_hook(hook_out))
        elif n.endswith(".post_attention_layernorm"):
            handles.append(m.register_forward_hook(hook_residual))

    for i, (prompt_label, prompt) in enumerate(zip(prompts_labels, prompts)):
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding=True).to(device)

        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            while input_ids.size(1) < (inputs["input_ids"].size(1) + 32):
                temp_activations = {'inp': [], 'out': [], "residual": []}
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                new_input_ids = []
                new_attention_mask = []
                n_ended = 0

                for j in range(input_ids.size(0)):
                    length_prediction = torch.min(torch.argwhere(torch.cat([torch.logical_not(attention_mask[j]), torch.tensor(True).to(device).unsqueeze(-1)])))
                    next_token_logits = outputs.logits[j, length_prediction-1, :]
                    next_token_id = torch.argmax(next_token_logits).unsqueeze(-1)

                    new_input_ids.append(torch.cat([input_ids[j,:length_prediction], next_token_id, input_ids[j,length_prediction:]], dim=-1))

                    if (next_token_id == tokenizer.eos_token_id).all():
                        n_ended += 1
                        next_token_attention = 0
                    else:
                        next_token_attention = 1
                    new_attention_mask.append(torch.cat([attention_mask[j,:length_prediction], torch.tensor(next_token_attention).to(device).unsqueeze(-1), attention_mask[j,length_prediction:]], dim=-1))
                
                input_ids = torch.stack(new_input_ids, dim=0)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                if n_ended == input_ids.size(0):
                    break
            attention_mask[:,-1] = torch.zeros((input_ids.size(0))).to(device)

        for k, v in temp_activations.items():
            temp_activations[k] = torch.stack(v, dim=0)

        activations[prompt_label]["inp"] = []
        activations[prompt_label]["out"] = []
        activations[prompt_label]["len"] = []
        for j in range(input_ids.size(0)):
            length_prompt = torch.min(torch.argwhere(torch.cat([torch.logical_not(inputs["attention_mask"][j]), torch.tensor(True).to(device).unsqueeze(-1)])))
            length_prediction = torch.min(torch.argwhere(torch.cat([torch.logical_not(attention_mask[j]), torch.tensor(True).to(device).unsqueeze(-1)])))
            if n_tok_start == n_tok_stop:
                if n_tok_stop == -6:
                    n_tok_activation_start = length_prompt - 1
                else:
                    n_tok_activation_start = length_prompt - n_tok_prompt[j]
                if n_tok_stop == -1 or n_tok_stop == -2 or n_tok_stop == -6:
                    n_tok_activation_stop = length_prediction
                elif n_tok_stop == -3 or n_tok_stop == -4:
                    n_tok_activation_stop = length_prompt
                else:
                    n_tok_activation_stop = n_tok_start + 1
            else:
                n_tok_activation_start = length_prompt + n_tok_start
                n_tok_activation_stop = length_prompt + n_tok_stop
            if prompt_label == "gld":
                print("Gold prompt:", tokenizer.decode(input_ids[j,:length_prediction]))
            else:
                print("Err prompt:", tokenizer.decode(input_ids[j, :length_prediction]))
            print("Editing:", tokenizer.decode(input_ids[j,n_tok_activation_start:n_tok_activation_stop]))
            
            activations[prompt_label]["inp"].append(temp_activations["inp"][:,j,n_tok_activation_start:n_tok_activation_stop])
            activations[prompt_label]["out"].append(temp_activations["out"][:,j,n_tok_activation_start:n_tok_activation_stop] + temp_activations["residual"][:,j,n_tok_activation_start:n_tok_activation_stop])
            activations[prompt_label]["len"].append((n_tok_activation_stop - n_tok_activation_start).cpu())

    for prompt_label in prompts_labels:
        activations[prompt_label]["inp"] = torch.cat(activations[prompt_label]["inp"], dim=1)
        activations[prompt_label]["out"] = torch.cat(activations[prompt_label]["out"], dim=1)

    for handle in handles:
        handle.remove()

    return activations

def get_collinearities(mat):
    distances = torch.cdist(mat, mat, p=2) / torch.sqrt(torch.tensor(mat.size(0)).to(device))
    print("mat", mat)
    print("distances", distances)
    plt.imshow(distances.cpu().detach().numpy(), vmin=distances.min(), vmax=distances.max())
    plt.colorbar()
    plt.title("distances")
    plt.savefig("distances.png")
    plt.close()
    
    is_close = torch.where(distances < 0.03, 1., 0.)
    is_close_norm = torch.div(is_close, torch.sum(is_close, dim=0)).T
    fused_closeness = torch.where(torch.isclose(torch.cdist(is_close_norm, is_close_norm, p=1), torch.tensor(2.0)), 0., 1.)
    collinearities = torch.flip(torch.unique(fused_closeness, dim=0), dims=(0,))
    return torch.div(collinearities.T, torch.sum(collinearities, dim=1)).T


def print_ranks(name, m):
    rank = torch.linalg.matrix_rank(m).detach().item()

    # frobenius_norm_squared = torch.sum(m**2)
    # spectral_norm_squared = torch.max(torch.linalg.svdvals(m))**2
    # stable_rank = (frobenius_norm_squared/spectral_norm_squared).detach().item()

    # singularvalues = torch.linalg.svdvals(m)
    # singularvalues = singularvalues ** 2
    # srank = (torch.sum(singularvalues, dim=-1) / torch.max(singularvalues, dim=-1).values)
    # stable_rank = srank.mean().item()

    mean_m = m.mean(dim=0)
    mmT = torch.matmul(m.T, m) / m.size(1) # Outer product averaged over N samples
    cov_m = mmT - torch.outer(mean_m, mean_m)
    print("cov and m", cov_m.size(), m.size())
    print("col", cov_m[895])
    print("col", cov_m[:,895])
    S = torch.linalg.svdvals(cov_m[:896, :896])

    # S = torch.linalg.svdvals(m)
    total_energy = torch.sum(S)
    cumulative_energy = torch.cumsum(S, dim=0) / total_energy
    stable_rank = torch.sum(cumulative_energy <= 0.9).item() + 1  # +1 due to zero indexing

    print(name, "| Rank:", rank, "| Stable rank:", stable_rank)
    print("SVD vals:", torch.round(torch.linalg.svdvals(m), decimals=4))
    return stable_rank

def modify_layers(model, layer_to_modify, insertion_type, activations, strength, treshold):
    activations_lengths = [min(gld_length, err_length) for gld_length, err_length in zip(activations["gld"]["len"], activations["err"]["len"])]
    print("Activations lengths", activations_lengths)
    print("Generalisation activations length", sum(activations_lengths[1:(len(activations_lengths)-1)//2 + 1]))
    print("Neighboor activations length", sum(activations_lengths[(len(activations_lengths)-1)//2 + 1:]))

    print("Activations cumsum", np.cumsum(activations_lengths))
    err_indices = np.concatenate([[0], np.cumsum(activations["err"]["len"])[:-1]])
    gld_indices = np.concatenate([[0], np.cumsum(activations["gld"]["len"])[:-1]])

    err_mask = torch.zeros(activations["err"]["inp"].size(1)).to(torch.bool)
    for start, stop in zip(err_indices, err_indices + activations_lengths):
        err_mask[start:stop] = True
    gld_mask = torch.zeros(activations["gld"]["inp"].size(1)).to(torch.bool)
    for start, stop in zip(gld_indices, gld_indices + activations_lengths):
        gld_mask[start:stop] = True

    err_inp = activations["err"]["inp"][:, err_mask].to(device)
    err_out = activations["err"]["out"][:, err_mask].to(device)
    gld_inp = activations["gld"]["inp"][:, gld_mask].to(device)
    gld_out = activations["gld"]["out"][:, gld_mask].to(device)


    n_layers = err_inp.size(0)
    n_tok = min(err_inp.size(1), gld_inp.size(1))
    d_model = err_inp.size(2)

    print("Vectors dims", err_inp.size(), err_out.size(), err_inp.size(), gld_inp.size(), gld_out.size(), gld_inp.size())

    # if insertion_type == "all":
    #     # not working yet, only use for debug purpose
    #     x = err_inp
    #     y = gld_out - err_out
    #     # wn = torch.tensor([m.weight for n,m in model.named_modules() if n.endswith(".post_attention_layernorm")]).to(device)
    #     # print("wn", wn.size(), x.size())
    #     norm_squared_x = torch.div(x, (torch.norm(x, dim=2).unsqueeze(-1)**2))
    #     # norm_squared_x = torch.div(x, wn.unsqueeze(1)**2)/wn.size(0)
    #     w_up = norm_squared_x/strength
    #     w_gate = strength*(norm_squared_x - treshold)
    #     norm_edit = torch.matmul(x, norm_squared_x.permute(0, 2, 1))
    #     z_edit = norm_edit/strength
    #     g_edit = strength*(norm_edit - treshold)

    #     collinearities = [get_collinearities(norm_edit_layer) for norm_edit_layer in norm_edit]
    #     x_co = [collinearities_layer@x_layer for collinearities_layer, x_layer in zip(collinearities, x)]
    #     w_up_co = [collinearities_layer@w_up_layer for collinearities_layer, w_up_layer in zip(collinearities, w_up)]
    #     w_gate_co = [collinearities_layer@w_gate_layer for collinearities_layer, w_gate_layer in zip(collinearities, w_gate)]
    #     y_co = [collinearities_layer@y_layer for collinearities_layer, y_layer in zip(collinearities, y)]
    #     z_edit_co = [collinearities_layer@z_edit_layer@collinearities_layer.T for collinearities_layer, z_edit_layer in zip(collinearities, z_edit)]
    #     g_edit_co = [collinearities_layer@g_edit_layer@collinearities_layer.T for collinearities_layer, g_edit_layer in zip(collinearities, g_edit)]
        
    #     # gated_z_edit = [z_edit_layer*z_edit_layer*torch.nn.functional.sigmoid(z_edit_layer) for z_edit_layer in z_edit]
    #     # gated_z_edit = [z_edit_layer*z_edit_layer*torch.nn.functional.sigmoid(strength*z_edit_layer) for z_edit_layer in z_edit]
    #     gated_z_edit = [z_edit_layer*g_edit_layer*torch.nn.functional.sigmoid(g_edit_layer) for z_edit_layer, g_edit_layer in zip(z_edit, g_edit)]
    #     gated_z_edit_co = [z_edit_layer*g_edit_layer*torch.nn.functional.sigmoid(g_edit_layer) for z_edit_layer, g_edit_layer in zip(z_edit_co, g_edit_co)]
    #     print("Collinearities", collinearities)
    #     print("Gated z edit", gated_z_edit)
    #     w_down = [torch.linalg.solve(gated_z_edit_layer, y_layer).T for gated_z_edit_layer, y_layer in zip(gated_z_edit_co, y_co)] + [y_co[-1].T]
    #     w_down = [torch.linalg.solve(gated_z_edit_layer, y_layer).T for gated_z_edit_layer, y_layer in zip(gated_z_edit_co, y_co)] + [y_co[-1].T]
    
    #     n_new_vecs = torch.max([256 * ((gated_z_edit_layer.size(0)+255)//256) for gated_z_edit_layer in gated_z_edit])
    # else:
    # for n,m in model.named_modules():
    #     if n.endswith(".post_attention_layernorm") and int(n.split(".")[2]) == layer_to_modify:
    #         wn = m.weight
    #         break

    x = err_inp[layer_to_modify]
    y = gld_out[layer_to_modify] - err_out[layer_to_modify]
    # y[680:] = torch.zeros((569, 896)).to(device)
    
    w_up = torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2))
    # w_up = x

    plt.imshow(x.cpu().detach().numpy(), vmin=x.min(), vmax=x.max())
    plt.colorbar()
    plt.title("x")
    plt.savefig("x.png")
    plt.close()

    plt.imshow(y.cpu().detach().numpy(), vmin=y.min(), vmax=y.max())
    plt.colorbar()
    plt.title("y")
    plt.savefig("y.png")
    plt.close()

    plt.imshow(w_up.cpu().detach().numpy(), vmin=w_up.min(), vmax=w_up.max())
    plt.colorbar()
    plt.title("w_up")
    plt.savefig("w_up.png")
    plt.close()

    # # Get principal components
    # u, s, vt = torch.linalg.svd(x)
    # uk = u.T[:]
    # x = torch.matmul(uk, x)
    # y = torch.matmul(uk, y)
    # w_up = torch.matmul(uk, w_up)
    # w_up = torch.div(w_up, (torch.norm(w_up, dim=1).unsqueeze(-1)**2))

    # svd = TruncatedSVD(n_components=x.size(0) - 1, algorithm="arpack")
    # x = torch.tensor(svd.fit_transform(x.T.cpu())).to(device).T
    # y = torch.tensor(svd.transform(y.T.cpu())).to(device).T
    # w_up = torch.tensor(svd.transform(w_up.T.cpu())).to(device).T
    # w_up = torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2))

    # w_up = torch.div(x, wn.unsqueeze(0)**2)/wn.size(0)

    # w_up = torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)))
    # w_up = torch.div(w_up, wn.unsqueeze(0))/torch.sqrt(torch.tensor(wn.size(0)).to(device))

    # x_norm = torch.norm(x, dim=1).unsqueeze(-1)
    # w_up = torch.div(x, wn.unsqueeze(0))/torch.sqrt(torch.tensor(wn.size(0)).to(device))
    # w_up = torch.div(w_up, x_norm)

    z_edit = torch.matmul(x, w_up.T)
    print("Z edit before collinearities", z_edit)    
    # print("diagonal", torch.diagonal(z_edit))
    # print("off diagonal min", torch.min(z_edit - 0.01*torch.eye(z_edit.size(0)).to(device)))
    # print("off diagonal max", torch.max(z_edit - 0.01*torch.eye(z_edit.size(0)).to(device)))
    plt.imshow(z_edit.cpu().detach().numpy(), vmin=z_edit.min(), vmax=z_edit.max())
    plt.colorbar()
    plt.title("z_edit_pre")
    plt.savefig("z_edit_pre.png")
    plt.close()

    torch.save(z_edit, "z_edit.pt")
    torch.save(y, "y.pt")

    # collinearities = get_collinearities(z_edit)
    # x = collinearities@x
    # w_up = [collinearities@w_up]
    # y = collinearities@y
    # z_edit = collinearities@z_edit@collinearities.T

    # u, s, vt = torch.linalg.svd(z_edit)
    # uk = u.T[:]
    # x = uk@x
    # w_up = uk@w_up
    # y = uk@y
    # # z_edit = uk@z_edit@uk.T
    # w_up = torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2))
    # z_edit = torch.matmul(x, w_up.T)
    w_up = [w_up]

    # svd = TruncatedSVD(n_components=z_edit.size(0) - 1, algorithm="arpack")
    # z_edit = torch.tensor(svd.fit_transform(z_edit.T.cpu())).to(device).T
    # print("mid zedit", z_edit.size())
    # z_edit = torch.tensor(svd.transform(z_edit.cpu())).to(device)
    # x = torch.tensor(svd.transform(x.T.cpu())).to(device).T
    # y = torch.tensor(svd.transform(y.T.cpu())).to(device).T
    # w_up = torch.tensor(svd.transform(w_up.T.cpu())).to(device).T
    # w_up = [torch.div(w_up, (torch.norm(x, dim=1).unsqueeze(-1)**2))]

    gated_z_edit = z_edit*(z_edit - treshold)*torch.nn.functional.sigmoid(strength*(z_edit - treshold))
    # gated_z_edit = (z_edit + (1 / ((1-treshold) * torch.nn.functional.sigmoid(torch.tensor(strength*(1-treshold))))) - 1)*(z_edit - treshold)*torch.nn.functional.sigmoid(strength*(z_edit - treshold))
    # gated_z_edit = (z_edit + BIAS_UP)*(z_edit - treshold)*torch.nn.functional.sigmoid(strength*(z_edit - treshold))
    print("Z edit", z_edit)    
    print("diagonal", torch.diagonal(z_edit))
    # print("off diagonal min", torch.min(z_edit - 0.01*torch.eye(z_edit.size(0)).to(device)))
    # print("off diagonal max", torch.max(z_edit - 0.01*torch.eye(z_edit.size(0)).to(device)))
    print("Default Condition number", torch.linalg.cond(gated_z_edit))
    # P = torch.diag(1.0 / torch.sqrt(torch.diag(gated_z_edit) + 1e-6))  # Avoid division by zero

    # u, s, v = torch.linalg.svd(gated_z_edit)
    # P = u.T[:] 
    # gated_z_edit = P @ gated_z_edit @ P.T
    # y = torch.matmul(P, y)

    best_lamb = 0
    best_mse = 10000000000
    for lamb in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        reg_gated_z_edit = gated_z_edit + lamb*torch.eye(gated_z_edit.size(0)).to(device)
        condition_number = torch.linalg.cond(reg_gated_z_edit)
        if condition_number < 1e9:
            mse = torch.mean((torch.matmul(gated_z_edit, torch.linalg.solve(reg_gated_z_edit, y)) - y) ** 2)
            print("Lamb", lamb, "MSE:", mse, "Condition number", condition_number)
            if mse < best_mse:
                best_mse = mse
                best_lamb = lamb

    print("Best lamb", best_lamb)

    gated_z_edit += best_lamb * torch.eye(gated_z_edit.size(0)).to(device)
    print("Deconditioned Condition number", torch.linalg.cond(gated_z_edit))

    w_down = [torch.linalg.solve(gated_z_edit, y).T]
    # w_down = [torch.matmul(P.T, w_down[0].T).T]

    n_new_vecs = 256 * ((gated_z_edit.size(0)+255)//256)

    # print("Collinearities", collinearities)
    print("Gated z edit", gated_z_edit)    
    print("diagonal", torch.diagonal(gated_z_edit))
    # print("off diagonal min", torch.min(gated_z_edit - 0.01*torch.eye(gated_z_edit.size(0)).to(device)))
    # print("off diagonal max", torch.max(gated_z_edit - 0.01*torch.eye(gated_z_edit.size(0)).to(device)))
    print("size", gated_z_edit.size())
    plt.imshow(gated_z_edit.cpu().detach().numpy(), vmin=gated_z_edit.min(), vmax=gated_z_edit.max())
    plt.colorbar()
    plt.title("gated_z_edit")
    plt.savefig("gated_z_edit.png")
    plt.close()

    plt.imshow(z_edit.cpu().detach().numpy(), vmin=z_edit.min(), vmax=z_edit.max())
    plt.colorbar()
    plt.title("z_edit")
    plt.savefig("z_edit.png")
    plt.close()

    plt.imshow(x.cpu().detach().numpy(), vmin=x.min(), vmax=x.max())
    plt.colorbar()
    plt.title("x_post")
    plt.savefig("x_post.png")
    plt.close()

    plt.imshow(y.cpu().detach().numpy(), vmin=y.min(), vmax=y.max())
    plt.colorbar()
    plt.title("y_post")
    plt.savefig("y_post.png")
    plt.close()

    plt.imshow(w_up[0].cpu().detach().numpy(), vmin=w_up[0].min(), vmax=w_up[0].max())
    plt.colorbar()
    plt.title("w_up_post")
    plt.savefig("w_up_post.png")
    plt.close()
        
    # l1 = torch.nn.L1Loss()
    # mse = torch.nn.MSELoss()
    # cosine_sim = torch.nn.CosineSimilarity(dim=1)
    
    # dataset = TensorDataset(x, y)
    # train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # config = Qwen2ModifiedConfig(
    #     hidden_act="silu",
    #     hidden_size=896,
    #     intermediate_size=125
    # )
    # edit_mlp = Qwen2MLP(config).to(device)
    # optimizer = torch.optim.Adam(edit_mlp.parameters(), lr=1e-4, weight_decay=1e-4)
    # print("Edit MLP", edit_mlp.up_proj.weight.size(), edit_mlp.gate_proj.weight.size(), edit_mlp.down_proj.weight.size())
    # best_model = edit_mlp
    # best_loss = 10000000000
    # best_lr = 1e-4

    # # for lr in [1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5, 5e-6, 3e-6, 1e-6]:
    # for lr in [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]:
    #     # for nn_size in [12, 25, 50, 125, 250, 500, 1000]:
    #     for nn_size in [12, 125, 1249, 2500]:
    #         config = Qwen2ModifiedConfig(
    #             hidden_act="silu",
    #             hidden_size=896,
    #             intermediate_size=nn_size
    #         )
    #         edit_mlp = Qwen2MLP(config).to(device)
    #         # edit_mlp.up_proj.weight = torch.nn.Parameter(1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)))
    #         # edit_mlp.gate_proj.weight = torch.nn.Parameter(1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)))
    #         # edit_mlp.down_proj.weight = torch.nn.Parameter(y.T)
    #         # edit_mlp.up_proj.weight = torch.nn.Parameter(w_up[0])
    #         # edit_mlp.gate_proj.weight = torch.nn.Parameter(w_up[0])
    #         # edit_mlp.down_proj.weight = torch.nn.Parameter(w_down[0])
    #         optimizer = torch.optim.Adam(edit_mlp.parameters(), lr=lr, weight_decay=1e-4)
    #         for epoch in range(1000):
    #             edit_mlp.train()

    #             running_loss = 0.0
    #             for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

    #                 predictions = edit_mlp(batch_x)
    #                 loss = mse(predictions, batch_y)
    #                 # loss = mse(predictions, batch_y) + 100000*torch.mean((edit_mlp.up_proj.weight-1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)))**2) + 100000*torch.mean((edit_mlp.gate_proj.weight-1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)))**2) + 1000*torch.mean((edit_mlp.down_proj.weight-y.T)**2)
    #                 # loss = l1(predictions, batch_y) - torch.log(torch.nn.functional.sigmoid(cosine_sim(predictions, batch_y))).mean()

    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(edit_mlp.parameters(), 1.0)
    #                 optimizer.step()

    #                 running_loss += loss.item()

    #             avg_loss = running_loss / len(train_loader)

    #             if avg_loss < best_loss:
    #                 best_loss = avg_loss
    #                 best_lr = lr
    #                 best_nn_size = nn_size
    #                 best_model_state_dict = edit_mlp.state_dict()

    #         print(f"LR {lr} NN SIZE {nn_size}, Loss: {avg_loss}")

    # # best_lr = 1e-4
    # # best_nn_size = int(x.size(0) * 0.1)
    # # best_nn_size = x.size(0)
    # n_new_vecs = best_nn_size + 1
    # print("Best LR:", best_lr)
    # print("Best NN size:", best_nn_size)
    # config = Qwen2ModifiedConfig(
    #     hidden_act="silu",
    #     hidden_size=896,
    #     intermediate_size=best_nn_size
    # )
    # edit_mlp = Qwen2MLP(config).to(device)
    # optimizer = torch.optim.Adam(edit_mlp.parameters(), lr=best_lr, weight_decay=1e-4)
    # print("Edit MLP", edit_mlp.up_proj.weight.size(), edit_mlp.gate_proj.weight.size(), edit_mlp.down_proj.weight.size())
    # best_model = edit_mlp
    # best_loss = 10000000000


    # # edit_mlp.up_proj.weight = torch.nn.Parameter(1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)), requires_grad=True)
    # # edit_mlp.gate_proj.weight = torch.nn.Parameter(1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)), requires_grad=True)
    # # edit_mlp.down_proj.weight = torch.nn.Parameter(y.T)
    # # edit_mlp.up_proj.weight = torch.nn.Parameter(w_up[0])
    # # edit_mlp.gate_proj.weight = torch.nn.Parameter(w_up[0])
    # # edit_mlp.down_proj.weight = torch.nn.Parameter(w_down[0])
    # # best_model_state_dict = edit_mlp.state_dict()

    # print("Edit MLP", edit_mlp.up_proj.weight.size(), edit_mlp.gate_proj.weight.size(), edit_mlp.down_proj.weight.size())

    # plt.hist(torch.norm(1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)), dim=1).cpu())
    # plt.savefig("x_norms.png")
    # plt.close()
    # plt.hist(torch.norm(y.T, dim=1).cpu())
    # plt.savefig("y_norms.png")
    # plt.close()

    # epochs = 1000
    # for epoch in range(epochs):
    #     edit_mlp.train()

    #     running_loss = 0.0
    #     for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

    #         predictions = edit_mlp(batch_x)
    #         # loss = l1(predictions, batch_y) + mse(predictions, batch_y) - torch.log(torch.nn.functional.sigmoid(cosine_sim(predictions, batch_y))).mean() 
    #         # loss = mse(predictions, batch_y)
    #         # loss = mse(predictions, batch_y) + 100000*torch.mean((edit_mlp.up_proj.weight-1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)))**2) + 100000*torch.mean((edit_mlp.gate_proj.weight-1.14*torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2)))**2) + 1000*torch.mean((edit_mlp.down_proj.weight-y.T)**2)
    #         loss = mse(predictions, batch_y)
    #         # loss = mse(predictions, batch_y) -torch.log(torch.nn.functional.sigmoid(cosine_sim(predictions, batch_y))).mean()
    #         # loss = l1(predictions, batch_y) - torch.log(torch.nn.functional.sigmoid(cosine_sim(predictions, batch_y))).mean()

    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(edit_mlp.parameters(), 1.0)
    #         optimizer.step()

    #         running_loss += loss.item()

    #     avg_loss = running_loss / len(train_loader)

    #     if avg_loss < best_loss:
    #         best_loss = avg_loss
    #         best_model_state_dict = edit_mlp.state_dict()   

    #     if epoch % 5 == 0:
    #         print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss}")

    # best_model = Qwen2MLP(config).to(device)
    # best_model.load_state_dict(best_model_state_dict)
    # best_model.eval()
    # plt.hist(torch.norm(best_model.up_proj.weight, dim=1).cpu().detach())
    # plt.savefig("w_up_norms.png")
    # plt.close()
    # plt.hist(torch.norm(best_model.gate_proj.weight, dim=1).cpu().detach())
    # plt.savefig("w_gate_norms.png")
    # plt.close()
    # plt.hist(torch.norm(best_model.down_proj.weight, dim=1).cpu().detach())
    # plt.savefig("w_down_norms.png")
    # plt.close()

    # w_up = [best_model.up_proj.weight]
    # w_gate = [best_model.gate_proj.weight]
    # w_down = [best_model.down_proj.weight]
    # w_up_bias = best_model.up_proj.bias
    # w_gate_bias = best_model.gate_proj.bias
    # print(w_up[0].size(), w_gate[0].size(), w_down[0].size(), w_up_bias.size(), w_gate_bias.size())
    # print("Norm diff:", torch.norm(best_model(x) - y))



    # torch.set_printoptions(sci_mode=False)
    # ranks = {"x": [[],[],[]], "w_up": [[],[],[]], "w_gate": [[],[],[]], "z_edit": [[],[],[]], "g_edit": [[],[],[]], "g_edit_prime": [[],[],[]], "gated_z_edit": [[],[],[]], "w_down": [[],[],[]], "y": [[],[],[]]}
    # for n,m in model.named_modules():
    #     if n.endswith(".mlp") or n.endswith(".mlp"):
    #         layer = int(n.split(".")[2])
    #         original_w_up = m.up_proj.weight
    #         original_w_down = m.down_proj.weight
    #         original_w_gate = m.gate_proj.weight
    #         print("\n\nLayer", layer, "-------------------------------------------------------------")

    #         print()
    #         ranks["x"][0].append(print_ranks("x", x[layer]))
    #         ranks["x"][1].append(print_ranks("x", x[layer]))
    #         ranks["x"][2].append(print_ranks("x", x[layer]))
    #         x_pca = PCA(n_components=2).fit_transform(x[layer].cpu().detach().numpy())
    #         print("pca shape", x_pca.shape)
    #         plt.scatter(x_pca[:11,0], x_pca[:11,1], label="Edit")
    #         plt.scatter(x_pca[11:64,0], x_pca[11:64,1], label="Generalisation")
    #         plt.scatter(x_pca[64:,0], x_pca[64:,1], label="Neighboor")
    #         plt.title("x")
    #         plt.legend()
    #         plt.savefig(f"x_pca_layer_{layer}.png")
    #         plt.close()

    #         print()
    #         ranks["w_up"][0].append(print_ranks("w_up", w_up[layer]))
    #         ranks["w_up"][1].append(print_ranks("original w_up", original_w_up))
    #         ranks["w_up"][2].append(print_ranks("combined w_up", torch.cat([original_w_up, w_up[layer]]).to(device)))

    #         print()
    #         ranks["w_gate"][0].append(print_ranks("w_gate", w_gate[layer]))
    #         ranks["w_gate"][1].append(print_ranks("original w_gate", original_w_gate))
    #         ranks["w_gate"][2].append(print_ranks("combined w_gate", torch.cat([original_w_gate, w_gate[layer]]).to(device)))
            
    #         print()
    #         original_z_edit = torch.matmul(x[layer], original_w_up.T)
    #         ranks["z_edit"][0].append(print_ranks("z_edit", z_edit[layer]))
    #         ranks["z_edit"][1].append(print_ranks("original z_edit", original_z_edit))
    #         ranks["z_edit"][2].append(print_ranks("combined z_edit", torch.cat([original_z_edit.T, z_edit[layer]]).to(device)))
    #         z_edit_pca = PCA(n_components=2).fit_transform(z_edit[layer].cpu().detach().numpy())
    #         print("pca shape", z_edit_pca.shape)
    #         plt.scatter(z_edit_pca[:11,0], z_edit_pca[:11,1], label="Edit")
    #         plt.scatter(z_edit_pca[11:64,0], z_edit_pca[11:64,1], label="Generalisation")
    #         plt.scatter(z_edit_pca[64:,0], z_edit_pca[64:,1], label="Neighboor")
    #         plt.title("z_edit")
    #         plt.legend()
    #         plt.savefig(f"z_edit_pca_layer_{layer}.png")
    #         plt.close()

    #         print()
    #         original_g_edit = torch.matmul(x[layer], original_w_gate.T)
    #         ranks["g_edit"][0].append(print_ranks("g_edit", g_edit[layer]))
    #         ranks["g_edit"][1].append(print_ranks("original g_edit", original_g_edit))
    #         ranks["g_edit"][2].append(print_ranks("combined g_edit", torch.cat([original_g_edit.T, g_edit[layer]]).to(device)))
    #         g_edit_pca = PCA(n_components=2).fit_transform(g_edit[layer].cpu().detach().numpy())
    #         print("pca shape", g_edit_pca.shape)
    #         plt.scatter(g_edit_pca[:11,0], g_edit_pca[:11,1], label="Edit")
    #         plt.scatter(g_edit_pca[11:64,0], g_edit_pca[11:64,1], label="Generalisation")
    #         plt.scatter(g_edit_pca[64:,0], g_edit_pca[64:,1], label="Neighboor")
    #         plt.title("g_edit")
    #         plt.legend()
    #         plt.savefig(f"g_edit_pca_layer_{layer}.png")
    #         plt.close()

    #         print()
    #         original_g_edit_prime = torch.nn.functional.sigmoid(original_g_edit)
    #         ranks["g_edit_prime"][0].append(print_ranks("g_edit_prime", torch.nn.functional.sigmoid(g_edit[layer])))
    #         ranks["g_edit_prime"][1].append(print_ranks("original g_edit_prime", original_g_edit_prime))
    #         ranks["g_edit_prime"][2].append(print_ranks("combined g_edit_prime", torch.cat([original_g_edit_prime.T, torch.nn.functional.sigmoid(g_edit[layer])]).to(device)))
    #         g_edit_prime_pca = PCA(n_components=2).fit_transform(torch.nn.functional.sigmoid(g_edit[layer]).cpu().detach().numpy())
    #         print("pca shape", g_edit_prime_pca.shape)
    #         plt.scatter(g_edit_prime_pca[:11,0], g_edit_prime_pca[:11,1], label="Edit")
    #         plt.scatter(g_edit_prime_pca[11:64,0], g_edit_prime_pca[11:64,1], label="Generalisation")
    #         plt.scatter(g_edit_prime_pca[64:,0], g_edit_prime_pca[64:,1], label="Neighboor")
    #         plt.title("g_edit_prime")
    #         plt.legend()
    #         plt.savefig(f"g_edit_prime_pca_layer_{layer}.png")
    #         plt.close()

    #         print()
    #         original_gated_z_edit = original_z_edit*original_g_edit_prime
    #         ranks["gated_z_edit"][0].append(print_ranks("gated_z_edit", gated_z_edit[layer]))
    #         ranks["gated_z_edit"][1].append(print_ranks("original gated_z_edit", original_gated_z_edit))
    #         ranks["gated_z_edit"][2].append(print_ranks("combined gated_z_edit", torch.cat([original_gated_z_edit.T, gated_z_edit[layer]]).to(device)))
    #         gated_z_edit_pca = PCA(n_components=2).fit_transform(gated_z_edit[layer].cpu().detach().numpy())
    #         print("pca shape", gated_z_edit_pca.shape)
    #         plt.scatter(gated_z_edit_pca[:11,0], gated_z_edit_pca[:11,1], label="Edit")
    #         plt.scatter(gated_z_edit_pca[11:64,0], gated_z_edit_pca[11:64,1], label="Generalisation")
    #         plt.scatter(gated_z_edit_pca[64:,0], gated_z_edit_pca[64:,1], label="Neighboor")
    #         plt.title("gated_z_edit")
    #         plt.legend()
    #         plt.savefig(f"gated_z_edit_pca_layer_{layer}.png")
    #         plt.close()

    #         print()
    #         ranks["w_down"][0].append(print_ranks("w_down", w_down[layer]))
    #         ranks["w_down"][1].append(print_ranks("original w_down", original_w_down))
    #         ranks["w_down"][2].append(print_ranks("combined w_down", torch.cat([original_w_down, w_down[layer]], dim=1).to(device)))

    #         print()
    #         original_y = torch.matmul(original_gated_z_edit, original_w_down.T)
    #         ranks["y"][0].append(print_ranks("y", y[layer]))
    #         ranks["y"][1].append(print_ranks("original y", original_y))
    #         ranks["y"][2].append(print_ranks("combined y", original_y + y[layer]))
    #         y_pca = PCA(n_components=2).fit_transform(y[layer].cpu().detach().numpy())
    #         print("pca shape", y_pca.shape)
    #         plt.scatter(y_pca[:11,0], y_pca[:11,1], label="Edit")
    #         plt.scatter(y_pca[11:64,0], y_pca[11:64,1], label="Generalisation")
    #         plt.scatter(y_pca[64:,0], y_pca[64:,1], label="Neighboor")
    #         plt.title("y")
    #         plt.legend()
    #         plt.savefig(f"y_pca_layer_{layer}.png")
    #         plt.close()

    #         if layer == 0:
    #             plt.imshow(torch.cat([original_w_up, w_up[layer]], dim=0).cpu().detach().numpy(), vmin=w_up[layer].min(), vmax=w_up[layer].max())
    #             print("w_up", original_w_up.mean(), w_up[layer].mean(), original_w_up.std(), w_up[layer].std())
    #             plt.colorbar()
    #             plt.title("w_up")
    #             plt.savefig("w_up.png")
    #             plt.close()

    #             plt.imshow(torch.cat([original_w_gate, w_gate[layer]], dim=0).cpu().detach().numpy(), vmin=w_gate[layer].min(), vmax=w_gate[layer].max())
    #             print("w_gate", original_w_gate.mean(), w_gate[layer].mean(), original_w_gate.std(), w_gate[layer].std())
    #             plt.colorbar()
    #             plt.title("w_gate")
    #             plt.savefig("w_gate.png")
    #             plt.close()

    #             plt.imshow(torch.cat([original_z_edit.T, z_edit[layer]], dim=0).cpu().detach().numpy(), vmin=z_edit[layer].min(), vmax=z_edit[layer].max())
    #             print("z_edit", original_z_edit.mean(), z_edit[layer].mean(), original_z_edit.std(), z_edit[layer].std())
    #             plt.colorbar()
    #             plt.title("z_edit")
    #             plt.savefig("z_edit.png")
    #             plt.close()

    #             plt.imshow(torch.cat([original_g_edit.T, g_edit[layer]], dim=0).cpu().detach().numpy(), vmin=g_edit[layer].min(), vmax=g_edit[layer].max())
    #             print("g_edit", original_g_edit.mean(), g_edit[layer].mean(), original_g_edit.std(), g_edit[layer].std())
    #             plt.colorbar()
    #             plt.title("g_edit")
    #             plt.savefig("g_edit.png")
    #             plt.close()

    #             plt.imshow(torch.cat([original_g_edit_prime.T, torch.nn.functional.sigmoid(g_edit[layer])], dim=0).cpu().detach().numpy(), vmin=torch.nn.functional.sigmoid(g_edit[layer])[layer].min(), vmax=torch.nn.functional.sigmoid(g_edit[layer])[layer].max())
    #             print("g_edit_prime", original_g_edit_prime.mean(), torch.nn.functional.sigmoid(g_edit[layer]).mean(), original_g_edit_prime.std(), torch.nn.functional.sigmoid(g_edit[layer]).std())
    #             plt.colorbar()
    #             plt.title("g_edit_prime")
    #             plt.savefig("g_edit_prime.png")
    #             plt.close()

    #             plt.imshow(torch.cat([original_gated_z_edit.T, gated_z_edit[layer]], dim=0).cpu().detach().numpy(), vmin=gated_z_edit[layer].min(), vmax=gated_z_edit[layer].max())
    #             print("gated_z_edit", original_gated_z_edit.mean(), gated_z_edit[layer].mean(), original_gated_z_edit.std(), gated_z_edit[layer].std())
    #             plt.colorbar()
    #             plt.title("gated_z_edit")
    #             plt.savefig("gated_z_edit.png")
    #             plt.close()

    #             plt.imshow(torch.cat([original_w_down, w_down[layer]], dim=1).cpu().detach().numpy(), vmin=original_w_down.min(), vmax=original_w_down.max())
    #             print("w_down", original_w_down.mean(), w_down[layer].mean(), original_w_down.std(), w_down[layer].std())
    #             plt.colorbar()
    #             plt.title("w_down")
    #             plt.savefig("w_down.png")
    #             plt.close()

    #             plt.imshow((original_y + y[layer]).cpu().detach().numpy(), vmin=(original_y + y[layer]).min(), vmax=(original_y + y[layer]).max())
    #             print("combined_y", (original_y + y[layer]).mean(), (original_y + y[layer]).std())
    #             plt.colorbar()
    #             plt.title("combined_y")
    #             plt.savefig("combined_y.png")
    #             plt.close()

    # for k, v in ranks.items():
    #     plt.plot(v[0], label="New")
    #     plt.plot(v[1], label="Original")
    #     plt.plot(v[2], label="Combined")
    #     plt.title(k)
    #     plt.legend()
    #     plt.savefig(f"{k}_rank.png")
    #     plt.close()

    for n,m in model.named_modules():
        if n.endswith(".mlp.gate_proj") or n.endswith(".mlp.up_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            b = m.bias

            if layer_to_modify == 0 or insertion_type != "reccursive":
                w = torch.cat([w, torch.zeros(n_new_vecs, w.size(1)).to(device)])
                b = torch.zeros(w.size(0)).to(device)
                m.in_features += n_new_vecs
            if layer == layer_to_modify or insertion_type == "all":
                if insertion_type == "all":
                    w_up_layer = w_up[layer]
                else:
                    # if n.endswith(".mlp.gate_proj"):
                    #     w_up_layer = w_gate[0]
                    # else:
                    w_up_layer = w_up[0]
                with torch.no_grad():
                    if n.endswith(".mlp.gate_proj"):
                        w[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = w_up_layer*strength
                        # w[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = w_up_layer*strength
                        # b[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = -treshold*strength
                        # b[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = w_gate_bias
                    else:
                        w[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = w_up_layer*(1/strength)
                        # w[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = w_up_layer*(1/strength)
                        # b[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = (1/strength)*((1 / ((1-treshold) * torch.nn.functional.sigmoid(torch.tensor(strength*(1-treshold))))) - 1)
                        # b[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = (1/strength)*BIAS_UP
                        # b[-n_new_vecs:-n_new_vecs+w_up_layer.size(0)] = w_up_bias
            m.weight = torch.nn.Parameter(w)
            m.bias = torch.nn.Parameter(b)
        elif n.endswith(".mlp.down_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer_to_modify == 0 or insertion_type != "reccursive":
                w = torch.cat([w,  torch.zeros(w.size(0), n_new_vecs).to(device)], dim=1)
                m.out_features += n_new_vecs
            if layer == layer_to_modify or insertion_type == "all":
                print("Modifying layer ", layer)
                if insertion_type == "all":
                    w_down_layer = w_down[layer]
                else:
                    w_down_layer = w_down[0]
                with torch.no_grad():
                    w[:,-n_new_vecs:-n_new_vecs+w_down_layer.size(1)] = w_down_layer
            m.weight = torch.nn.Parameter(w)
            # print(n, m.weight.size())

    if layer_to_modify == 0 or insertion_type!="reccursive":
        model.config.intermediate_size = model.config.intermediate_size + n_new_vecs

    return model

def main(model, tokenizer, gld_prompt, err_prompt, n_tok_prompt, n_tok_start, n_tok_stop, insertion_type, layer_to_modify, strength, treshold):
    prompts = [gld_prompt, err_prompt]
    prompts_labels = ["gld", "err"]
    activations = {"gld": {}, "err": {}}
    activations = extract_activations(model, tokenizer, prompts, prompts_labels, activations, n_tok_prompt, n_tok_start, n_tok_stop)

    if insertion_type == "reccursive":
        for i in range(layer_to_modify, model.config.num_hidden_layers):
            model = modify_layers(model, i, insertion_type, activations, strength, treshold)
            activations = extract_activations(model, tokenizer, [err_prompt], ["err"], activations, n_tok_prompt, n_tok_start, n_tok_stop)
    elif insertion_type == "single" or insertion_type == "all":
        model = modify_layers(model, layer_to_modify, insertion_type, activations, strength, treshold)
        activations = extract_activations(model, tokenizer, [err_prompt], ["err"], activations, n_tok_prompt, n_tok_start, n_tok_stop)
    else:
        raise Exception("Insertion type must be single, reccursive or all")

    return model, tokenizer

if __name__ == '__main__':
    model = sys.argv[1]
    tokenizer = sys.argv[2]
    gld_prompt = sys.argv[3]
    err_prompt = sys.argv[4]
    n_tok_prompt = int(sys.argv[5])
    n_tok_start = int(sys.argv[6])
    n_tok_stop = int(sys.argv[7])
    insertion_type = sys.argv[8]
    layer_to_modify = int(sys.argv[9])
    strength = float(sys.argv[10])
    treshold = float(sys.argv[11])
    main(model, tokenizer, gld_prompt, err_prompt, n_tok_prompt, n_tok_start, n_tok_stop, insertion_type, layer_to_modify, strength, treshold)