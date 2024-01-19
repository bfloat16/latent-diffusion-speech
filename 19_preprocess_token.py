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

rich_progress = Progress(TextColumn("Preprocess:"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

@torch.no_grad()
def preprocess(rank, units_path, model,in_dir, out_dir, num_workers, units_quantize_type="kmeans"):
    with rich_progress:
        units_path = units_path[rank::num_workers]
        task_id = rich_progress.add_task(f"rank:{rank}", total=len(units_path))
        if units_quantize_type == "vq":
            model = model.to(f"cuda:{rank%num_workers}")
        for unit_path in units_path:
            if units_quantize_type == "kmeans":
                unit = np.load(unit_path)
                token = cluster.get_cluster_result(model, unit)
                out_path = os.path.join(out_dir, os.path.basename(os.path.dirname(unit_path)), os.path.basename(unit_path))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, token)
            elif units_quantize_type == "vq":
                unit = torch.from_numpy(np.load(unit_path)).to(f"cuda:{rank%num_workers}")[None,:]
                _, token, _ = model(unit)
                token = token[0].detach().cpu().numpy()
                out_path = os.path.join(out_dir, os.path.basename(os.path.dirname(unit_path)), os.path.basename(unit_path))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, token)
            rich_progress.update(task_id, advance=1)

def main(in_dir, units_quantize_type, model, num_workers=1):
    out_dir = os.path.join(in_dir, "semantic_token")
    os.makedirs(out_dir, exist_ok=True)
    units_dir = os.path.join(in_dir, "units")
    filelist = glob(f"{units_dir}/**/*.npy", recursive=True)
    mp.spawn(preprocess, args=(filelist, model,in_dir, out_dir, num_workers, units_quantize_type), nprocs=num_workers, join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='configs/config.yaml')
    parser.add_argument("-n", "--num_workers", type=int, default=2)
    cmd =parser.parse_args()
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
    
    main(train_path, units_quantize_type, model, num_workers=num_workers)
    main(valid_path, units_quantize_type, model, num_workers=num_workers)