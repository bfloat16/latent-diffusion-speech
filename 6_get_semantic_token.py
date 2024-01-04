import os
import torch
import cluster
import numpy as np 
import torch.multiprocessing as mp
import argparse
from glob import glob
from logger import utils
from tools.tools import get_encdoer_out_channels
from vector_quantize_pytorch import VectorQuantize
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

rich_progress = Progress(
    TextColumn("Preprocess:"),
    BarColumn(bar_width=80), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    transient=True
    )

@torch.no_grad()
def preprocess_utterance(rank, units_path, model,in_dir, out_dir, num_workers, units_quantize_type="kmeans"):
    with rich_progress:
        task_id = rich_progress.add_task(f"rank:{rank}", total=len(units_path))
        units_path = units_path[rank::num_workers]
        if units_quantize_type == "vq":
            model = model.to(f"cuda:{rank%num_workers}")
        for unit_path in units_path:
            if units_quantize_type == "kmeans":
                unit = np.load(os.path.join(in_dir, "units" , unit_path))
                token = cluster.get_cluster_result(model, unit)
                out_path = os.path.join(out_dir, unit_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, token)
            elif units_quantize_type == "vq":
                unit = torch.from_numpy(np.load(os.path.join(in_dir, "units" , unit_path))).to(f"cuda:{rank%num_workers}")[None,:]
                _, token, _ = model(unit)
                token = token[0].detach().cpu().numpy()
                out_path = os.path.join(out_dir, unit_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, token)
        rich_progress.update(task_id, advance=1)

def preprocess(in_dir, units_quantize_type, model, num_workers=1):
    out_dir = os.path.join(in_dir, "semantic_token")
    os.makedirs(out_dir, exist_ok=True)
    units_dir = os.path.join(in_dir, "units")
    filelist = glob(f"{units_dir}/**/*.npy", recursive=True)
    
    mp.spawn(preprocess_utterance, args=(filelist, model,in_dir, out_dir, num_workers, units_quantize_type), nprocs=num_workers, join=True)
    
def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml')
    parser.add_argument("-n", "--num_workers", type=int, default=10)
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == "__main__":
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    num_workers = cmd.num_workers

    train_path = args['data']['train_path']
    valid_path = args['data']['valid_path']
    units_quantize_type = args['text2semantic']['train']['units_quantize_type']
    codebook_path = args['text2semantic']['model']['codebook_path']

    if units_quantize_type == "kmeans":
        model = cluster.get_cluster_model(codebook_path)

    elif units_quantize_type == "vq":
        model = VectorQuantize(
                dim = get_encdoer_out_channels(args['data']['encoder']),
                codebook_size = args['text2semantic']['model']['semantic_kmeans_num'],
                decay = 0.8,             
                commitment_weight = 1.0
            )
        model_para = torch.load(codebook_path)
        model.load_state_dict(model_para["model"])
    else:
        raise ValueError('[x] Unknown quantize_type: ' + units_quantize_type)
    
    preprocess(train_path, units_quantize_type, model, num_workers=num_workers)
    preprocess(valid_path, units_quantize_type, model, num_workers=num_workers)