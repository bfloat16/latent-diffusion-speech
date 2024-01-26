import os
import torch
import argparse
from tools.infer_tools import DiffusionSVC
from text2semantic.utils import get_language_model
from tools.tools import units_forced_alignment
import yaml
from tools.tools import DotDict
from text.cleaner import text_to_sequence
from cluster import get_cluster_model
import soundfile as sf
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from tools.tools import get_encdoer_out_channels

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-dm", "--diffusion_model", type=str, default='exp/diffusion/model_380000.pt')
    parser.add_argument("-lm", "--language_model",  type=str, default='exp/lm/model_1000.pt')
    parser.add_argument("-i",  "--input",           type=str, default="你说的对，但是原神是由米哈游自主研发的一款全新开放世界冒险游戏。")
    parser.add_argument("-d" , "--device",          type=str, default=None)
    parser.add_argument("-o",  "--output",          type=str, default='1.wav')
    parser.add_argument("-s",  "--speedup",         type=str, default=10)
    parser.add_argument("-me", "--method",          type=str, default='dpm-solver')
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    with torch.no_grad():
        cmd = parse_args()
        
        device = cmd.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        diffusion_svc = DiffusionSVC(device=device)
        diffusion_svc.load_model(model_path=cmd.diffusion_model, f0_max=800, f0_min=65)
        config_file = os.path.join(os.path.split(cmd.language_model)[0], 'config.yaml')
        with open(config_file, "r") as config:
            args = yaml.safe_load(config)
        args = DotDict(args)

        if args['text2semantic']['train']['units_quantize_type'] == "kmeans":
            codebook = get_cluster_model(args['text2semantic']['model']['codebook_path'])
            codebook = codebook.__dict__["cluster_centers_"]
            
            semantic_embedding = torch.nn.Embedding(
                codebook.shape[0],
                codebook.shape[1],
                _freeze = True
                )
            semantic_embedding.weight.data = torch.from_numpy(codebook)
            semantic_embedding.to(device)
        elif args['text2semantic']['train']['units_quantize_type'] == "vq":
            semantic_embedding = VectorQuantize(
                    dim = get_encdoer_out_channels(args['data']['encoder']),
                    codebook_size = args['text2semantic']['model']['semantic_kmeans_num'],
                    decay = 0.8,             
                    commitment_weight = 1.,
                    freeze_codebook=True,
                    use_cosine_sim=True,
                    codebook_dim = 32
                )
            model_para = torch.load(args['text2semantic']['model']['codebook_path'])
            semantic_embedding.load_state_dict(model_para["model"])
            semantic_embedding = semantic_embedding.to(device)
        else:
            raise ValueError(' [x] Unknown quantize_type: ' + args['text2semantic']['train']['units_quantize_type'])

        lm = get_language_model(**args).to(device)
        lm.load_state_dict(torch.load(cmd.language_model, map_location=torch.device(device))["model"])
        lm.eval()

        text = cmd.input
        spk_id = cmd.spk_id
        speedup = cmd.speedup
        method = cmd.method
        
        (phones, tones, lang_ids), (norm_text, word2ph) = text_to_sequence(text, 'ZH')
        
        phones, tones = torch.from_numpy(np.array(phones)).unsqueeze(0).long().to(device), torch.from_numpy(np.array(tones)).long().unsqueeze(0).to(device)
        
        spk_id_seq = torch.ones_like(phones) * spk_id
        semantic_token = lm.generate(phones,
                            tones,
                            attention_mask=None,
                            use_cache=None,
                            max_length=1024,
                            do_sample=True,
                            temperature=1.0,
                            top_k=5,
                            top_p=1.0,
                            repetition_penalty=1.0,
                            num_beams=1,
                            no_repeat_ngram_size = 0,
                            early_stopping = True,
                            spk_id = spk_id_seq,
                            end_gate_threshold = None
                            )

        if semantic_token[:,-1] == lm.semantic_eos_token_id:
            semantic_token = semantic_token[:,1:-1]
        else:
            semantic_token = semantic_token[:,1:]

        if args['text2semantic']['train']['units_quantize_type'] == "kmeans":
            semantic_emb = semantic_embedding(semantic_token)
        elif args['text2semantic']['train']['units_quantize_type'] == "vq":
            semantic_emb = semantic_embedding.get_codes_from_indices(semantic_token)
            semantic_emb = units_forced_alignment(semantic_emb,
                                                  scale_factor=(diffusion_svc.args['data']['sampling_rate']/diffusion_svc.args['data']['block_size'])/(diffusion_svc.args.data.encoder_sample_rate/args.data.encoder_hop_size))

        wav = diffusion_svc.infer(semantic_emb,f0=None,volume=None, spk_id = spk_id, infer_speedup=speedup, method=method)
        
        sf.write(cmd.output, wav.detach().cpu().numpy()[0,0], diffusion_svc.args['data']['sampling_rate'])