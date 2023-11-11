import os
import torch
import librosa
from text2semantic.saver import Saver,Saver_empty
from cluster import get_cluster_model
from ..utils import get_topk_acc
from tools.tools import clip_grad_value_
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import warnings
warnings.filterwarnings("ignore")

progress = Progress(
    TextColumn("Running: "),
    BarColumn(bar_width=80), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    "•",
    TextColumn("[progress.description]{task.description}"),
    transient=True
    )

@torch.no_grad()
def test(args, model, loader_test, diffusion_model, saver,semantic_embedding, accelerator):
    model.eval()

    test_loss = 0.
    topk_acc = 0
    num_batches = len(loader_test)

    with torch.no_grad():
        test_task = progress.add_task("Test", total=num_batches)
        for _, data in enumerate(loader_test):
            fn = data['name'][0]
            progress.update(test_task, description=f"audio={fn}")

            for k in data.keys():
                if type(data[k]) is torch.Tensor:
                    data[k] = data[k].to(accelerator.device)

            semantic_token = model.generate(
                phone = data["phone"],
                tone = data["tone"],
                attention_mask = data["encoder_attention_mask"],
                spk_id = data["spk_id"],
            )
            
            if semantic_token[:,-1] == model.semantic_eos_token_id:
                semantic_token = semantic_token[:,1:-1]
            else:
                semantic_token = semantic_token[:,1:]

            if args.train.units_quantize_type == "kmeans":
                semantic_emb = semantic_embedding(semantic_token)
            elif args.train.units_quantize_type == "vq":
                semantic_emb = semantic_embedding.get_codes_from_indices(semantic_token)

            if diffusion_model is not None:
                signal = diffusion_model.infer(semantic_emb, None, None, use_tqdm=False)
            else:
                signal = None

            result = model(**data)
            test_loss += result.loss.item()
            topk_acc += get_topk_acc(data["semantic"][0][1:], result.logits[0][:-1,:], k = 5)
            
            if signal is not None:
                path_audio = os.path.join(args.data.valid_path, 'audio', data['name'][0].replace(".npy",""))
                audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
                saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})
            progress.update(test_task, advance=1)

    test_loss /= num_batches
    topk_acc /= num_batches
    progress.remove_task(test_task)
    return test_loss, topk_acc

def train(args, initial_global_step, model, optimizer, scheduler, diffusion_model, loader_train, loader_valid, accelerator):
        # saver
    if accelerator.is_main_process:
        saver = Saver(args, initial_global_step=initial_global_step)
    else:
        saver = Saver_empty(args, initial_global_step=initial_global_step)
    
    clip_grad_norm = float(args.model.text2semantic.train.clip_grad_norm) if args.model.text2semantic.train.clip_grad_norm != -1 else None

    if args.train.units_quantize_type == "kmeans":
        codebook = get_cluster_model(args.model.text2semantic.codebook_path)
        codebook = codebook.__dict__["cluster_centers_"]
        
        semantic_embedding = torch.nn.Embedding(
            codebook.shape[0],
            codebook.shape[1],
            _freeze = True
            )
        semantic_embedding.weight.data = torch.from_numpy(codebook)
        semantic_embedding.to(accelerator.device)
    elif args.train.units_quantize_type == "vq":
        from vector_quantize_pytorch import VectorQuantize
        semantic_embedding = VectorQuantize(
                dim = args.data.encoder_out_channels,
                codebook_size = args.model.text2semantic.semantic_kmeans_num,
                decay = 0.8,             
                commitment_weight = 1.,
                freeze_codebook=True
            )
        model_para = torch.load(args.model.text2semantic.codebook_path)
        semantic_embedding.load_state_dict(model_para["model"])
        semantic_embedding = semantic_embedding.to(accelerator.device)
    else:
        raise ValueError('[x] Unknown quantize_type: ' + args.train.units_quantize_type)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    with progress:
        train_task = progress.add_task("Train", total=num_batches - 1)
        for epoch in range(start_epoch, args.model.text2semantic.train.epochs):
            for _, data in enumerate(loader_train):
                with accelerator.accumulate(model):
                    if accelerator.sync_gradients:
                        saver.global_step_increment()

                    optimizer.zero_grad()

                    for k in data.keys():
                        if type(data[k]) is torch.Tensor:
                            data[k] = data[k].to(accelerator.device)
                            if k == "phone":
                                data[k][data[k] == -100] = accelerator.unwrap_model(model).PAD
                            if k == "tone":
                                data[k][data[k] == -100] = accelerator.unwrap_model(model).num_tones
                            if k == "semantic":
                                data[k][data[k] == -100] = accelerator.unwrap_model(model).semantic_pad_token_id

                    loss = model(**data).loss
                    grad_norm = clip_grad_value_(model.parameters(), clip_grad_norm)

                    loss += grad_norm

                    if torch.isnan(loss):
                        raise ValueError('[x] nan loss ')
                    else:
                        accelerator.backward(loss)
                        optimizer.step()
                        scheduler.step()
                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    progress.update(train_task, advance=1, description=f"epoch={epoch}, step={saver.global_step}, lr={current_lr:.7f}, loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

                if accelerator.is_main_process and saver.global_step % args.train.interval_log == 0:
                    saver.log_value({'train/loss': loss.item()})
                    saver.log_value({'train/lr': current_lr})

                if accelerator.is_main_process and saver.global_step % args.train.interval_val == 0:
                    optimizer_save = optimizer if args.model.text2semantic.train.save_opt else None
                    unwrap_model = accelerator.unwrap_model(model)

                    if saver.global_step % args.train.interval_force_save == 0:
                        saver.save_model(unwrap_model, optimizer_save, postfix=f'{saver.global_step}_Force')
                    else:
                        saver.save_model(unwrap_model, optimizer, postfix=f'{saver.global_step}')

                    last_val_step = saver.global_step - args.train.interval_val * (args.train.last_save_model_num + 1)
                    saver.delete_model(postfix=f'{last_val_step}')

                    test_loss, topk_acc = test(args, unwrap_model, loader_valid, diffusion_model, saver,semantic_embedding, accelerator)

                    saver.log_value({'val/loss': test_loss})
                    saver.log_value({'val/top_acc@5': topk_acc})

                    model.train()
                accelerator.wait_for_everyone()
            progress.reset(train_task)