import os
import torch
import librosa
from logger.saver import Saver, Saver_empty
from tools.tools import clip_grad_value_
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

progress = Progress(TextColumn("Running: "), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn(), 
                    "•", TextColumn("[progress.description]{task.description}"))

def test(args, model, vocoder, loader_test, f0_extractor, quantizer, saver, accelerator):
    model.eval()

    test_loss = 0.
    num_batches = len(loader_test)

    with torch.no_grad():
        utilization = 0
        count = torch.zeros(args.model.text2semantic.semantic_kmeans_num).to(accelerator.device)
        test_task = progress.add_task("Test", total=num_batches)
        for _, data in enumerate(loader_test):
            fn = data['name'][0]
            progress.update(test_task, description=f"audio={fn}")
            
            if args['diffusion']['model']['is_tts']:
                data['f0'] = None
            if args['diffusion']['model']['is_tts']:
                data['volume'] = None
            if args['diffusion']['model']['is_tts']:
                data['aug_shift'] = None

            for k in data.keys():
                if type(data[k]) is torch.Tensor:
                    data[k] = data[k].to(accelerator.device)

            if quantizer is not None:
                if args['text2semantic']['train']['units_quantize_type'] == "kmeans":
                    data['units'] = quantizer(data['units']).detach()
                    commit_loss = 0
                elif args['text2semantic']['train']['units_quantize_type'] == "vq":
                    data['units'], indices, commit_loss = quantizer(data['units'])
                    count += torch.bincount(indices.flatten(), minlength=args.model.text2semantic.semantic_kmeans_num)
                else:
                    raise ValueError('[x] Unknown quantize_type: ' + args['text2semantic']['train']['units_quantize_type'])
            else:
                commit_loss = 0

            mel = model(
                data['units'],
                data['f0'],
                data['volume'],
                data['spk_id'],
                gt_spec=data['mel'],
                infer=True,
                infer_speedup=args['common']['infer']['speedup'],
                method=args['common']['infer']['method'],
                )
            
            if data['f0'] is None:
                f0 = f0_extractor.model(mel=mel, infer=True, return_hz_f0=True)[:,:,0]
            else:
                f0 = data['f0']

            signal = vocoder.infer(mel, f0)

            loss = model(
                data['units'],
                data['f0'],
                data['volume'],
                data['spk_id'],
                gt_spec=data['mel'],
                infer=False
                )
            
            test_loss += loss.item()
            test_loss += commit_loss

            saver.log_spec(data['name'][0], data['mel'], mel)

            path_audio = os.path.join(args['data']['valid_path'], 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args['data']['sampling_rate'])
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})
            progress.update(test_task, advance=1)

    test_loss /= num_batches
    utilization = torch.sum(count > 0).item() / args.model.text2semantic.semantic_kmeans_num
    test_loss = test_loss.item()
    saver.log_value({'valid/utilization': utilization})
    progress.remove_task(test_task)
    return test_loss

def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test,quantizer, accelerator):
    if accelerator.is_main_process:
        saver = Saver(args, initial_global_step=initial_global_step)
    else:
        saver = Saver_empty(args, initial_global_step=initial_global_step)

    clip_grad_norm = float(args['diffusion']['train']['clip_grad_norm']) if args['diffusion']['train']['clip_grad_norm'] != -1 else None

    device = accelerator.device

    if args['diffusion']['model']['is_tts']:
        from encoder.fcpe.model import FCPEInfer
        f0_extractor = FCPEInfer(model_path='pretrain/fcpe.pt')
    else:
        f0_extractor = None

    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    with progress:
        train_task = progress.add_task("Train", total=num_batches - 1)
        for epoch in range(start_epoch, args['diffusion']['train']['epochs']):
            for _, data in enumerate(loader_train):
                with accelerator.accumulate(model):
                    if args['diffusion']['model']['is_tts']:
                        data['f0'] = None
                    if args['diffusion']['model']['is_tts']:
                        data['volume'] = None
                    if args['diffusion']['model']['is_tts']:
                        data['aug_shift'] = None
                    if accelerator.sync_gradients:
                        saver.global_step_increment()

                    optimizer.zero_grad()

                    for k in data.keys():
                        if type(data[k]) is torch.Tensor:
                            data[k] = data[k].to(device)

                    if quantizer is not None:
                        if args['text2semantic']['train']['units_quantize_type'] == "kmeans":
                            data['units'] = quantizer(data['units']).detach()
                            commit_loss = 0
                        elif args['text2semantic']['train']['units_quantize_type'] == "vq":
                            data['units'], indices, commit_loss = quantizer(data['units'])
                        else:
                            raise ValueError('[x] Unknown quantize_type: ' + args['text2semantic']['train']['units_quantize_type'])
                    else:
                        commit_loss = 0

                    loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], aug_shift=data['aug_shift'], gt_spec=data['mel'].float(), infer=False) + commit_loss
                    
                    accelerator.backward(loss)
                    grad_norm = clip_grad_value_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    scheduler.step()
                
                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    vq_loss = commit_loss.item() if isinstance(commit_loss, torch.Tensor) else 0
                    progress.update(train_task, advance=1, description=f"epoch={epoch}, step={saver.global_step}, lr={current_lr:.7f}, loss={loss.item():.4f}, vq_loss={vq_loss:.4f}, grad_norm={grad_norm:.4f}")

                if accelerator.is_main_process and saver.global_step % args['diffusion']['train']['interval_log'] == 0:
                    saver.log_value({'train/loss': loss.item()})
                    saver.log_value({'train/vq_loss': commit_loss.item() if type(commit_loss) is torch.Tensor else 0})
                    saver.log_value({'train/grad_norm': grad_norm})
                    saver.log_value({'train/lr': current_lr})

                if accelerator.is_main_process and saver.global_step % args['diffusion']['train']['interval_val'] == 0:
                    if args['text2semantic']['train']['units_quantize_type'] == "vq":
                        saver.save_model(quantizer, None, postfix=f'{saver.global_step}_semantic_codebook')

                    unwrap_model = accelerator.unwrap_model(model)
                    test_loss = test(args, unwrap_model, vocoder, loader_test, f0_extractor, quantizer, saver, accelerator)
                    saver.log_value({'val/loss': test_loss})
                    saver.save_model(unwrap_model, optimizer, postfix=f'{saver.global_step}')
                    model.train()
                accelerator.wait_for_everyone()
            progress.reset(train_task)