import os
import torch
import shutil
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from kmeans import KMeansGPU
from sklearn.cluster import KMeans, MiniBatchKMeans

def train_cluster(dataset, n_clusters, use_minibatch=True, verbose=False,use_gpu=False):
    filenames = glob(f"{dataset}/*/*.wav.npy", recursive=True)
    shuffle(filenames)
    selected_filenames = filenames[:20000]

    if os.path.exists(tempset):
        shutil.rmtree(tempset)
    os.makedirs(tempset, exist_ok=True)

    for path in tqdm(selected_filenames):
        not_processed_feature = np.load(path)
        not_processed_feature = not_processed_feature.astype(np.float32)
        save_path = os.path.join(tempset, os.path.basename(path)) 
        np.save(save_path, not_processed_feature)
    filename = glob(f"{tempset}/*.wav.npy", recursive=True)

    features = []
    for path in tqdm(filename):
        features.append(np.load(path))
    features = np.concatenate(features, axis=0)

    print('Start KMeans')
    if (use_gpu is False):
        if use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, verbose=verbose, batch_size=4096, max_iter=80).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters, verbose=verbose).fit(features)
    else:
        kmeans = KMeansGPU(n_clusters=n_clusters, mode='euclidean', verbose=2 if verbose else 0,max_iter=500,tol=1e-2)
        features=torch.from_numpy(features)
        kmeans.fit_predict(features)

    x = {
        "n_features_in_": kmeans.n_features_in_ if use_gpu is False else features.shape[1],
        "_n_threads": kmeans._n_threads if use_gpu is False else 4,
        "cluster_centers_": kmeans.cluster_centers_ if use_gpu is False else kmeans.centroids.cpu().numpy()
    }
    print("End KMeans")
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default="./data/train/units")
    parser.add_argument('--tempset', type=Path, default="./cluster/temp")
    parser.add_argument('--output', type=Path, default="pretrain")
    parser.add_argument('--gpu',action='store_true', default=True)
    parser.add_argument('--n_clusters', type=int, default=2048)

    args = parser.parse_args()

    global tempset
    checkpoint_dir = args.output
    dataset = args.dataset
    tempset = args.tempset
    use_gpu = args.gpu
    n_clusters = args.n_clusters

    x = train_cluster(dataset, n_clusters, use_minibatch=False, verbose=False, use_gpu=use_gpu)

    checkpoint_path = checkpoint_dir / "semantic_codebook.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(x, checkpoint_path)