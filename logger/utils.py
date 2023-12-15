import os
import yaml
import json
import torch

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        # all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info[model_name] = trainable_params
    return info

def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args

def to_json(path_params, path_json):
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict = {}
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val

    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile, indent="\t")

def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

def load_model(
        expdir,
        model,
        optimizer,
        name='model',
        postfix='',
        device='cpu'):
    if postfix == '':
        postfix = '_' + postfix
    path = os.path.join(expdir, name + postfix)
    path_pt = traverse_dir(expdir, ['pt'], is_ext=False)
    global_step = 0
    if len(path_pt) > 0:
        steps = [s[len(path):] for s in path_pt]
        maxstep = max([int(s) if s.isdigit() else 0 for s in steps])
        if maxstep >= 0:
            path_pt = path + str(maxstep) + '.pt'
        else:
            path_pt = path + 'best.pt'
        print('restoring model from', path_pt)
        ckpt = torch.load(path_pt, map_location=torch.device(device))
        global_step = ckpt['global_step']
        model.load_state_dict(ckpt['model'], strict=False)
        if ckpt.get('optimizer') != None:
            optimizer.load_state_dict(ckpt['optimizer'])
    return global_step, model, optimizer
