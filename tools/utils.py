import os
import yaml
import torch

def traverse_dir(root_dir, extensions, amount=None, str_include=None, str_exclude=None, is_pure=False, is_sort=False, is_ext=True, second_root_dir=None):
    if second_root_dir == None:
        second_root_dir = root_dir
        
    file_list = []
    cnt = 0

    if not os.path.exists(root_dir):
        return file_list
    
    for file in os.scandir(root_dir):
        if file.is_file():
            if any([file.path.endswith(f".{ext}") for ext in extensions]):
                mix_path = file.path
                pure_path = mix_path[len(second_root_dir) + 1:] if is_pure else mix_path
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
        if file.is_dir():
            file_list += traverse_dir(file.path, extensions, amount, str_include, str_exclude, is_pure, is_sort, is_ext, root_dir)
    if is_sort:
        file_list.sort()
    return file_list

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info[model_name] = trainable_params
    return info

def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    return args

def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

def load_model(expdir, model, optimizer, name='model', postfix='', device='cpu'):
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