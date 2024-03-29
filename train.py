import argparse
import json
import os
import torch
# from utils import build_model
from utils import build_model, get_lr, average_checkpoints, last_n_checkpoints
import time
import gc
import numpy as np

from torch.utils.data import DataLoader
from models.loss import WaveFlowLossDataParallel
from mel2samp import Mel2Samp, MAX_WAV_VALUE
from scipy.io.wavfile import write

def stft(y, scale='linear'):
    D = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024)#, window=torch.hann_window(1024).cuda())
    D = torch.sqrt(D.pow(2).sum(-1) + 1e-10)
    # D = torch.sqrt(torch.clamp(D.pow(2).sum(-1), min=1e-10))
    if scale == 'linear':
        return D
    elif scale == 'log':
        S = 2 * torch.log(torch.clamp(D, 1e-10, float("inf")))
        return S
    else:
        pass

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']

    if optimizer is not None and scheduler is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])

    model_for_loading = checkpoint_dict['model']
    try:
        model.load_state_dict(model_for_loading)
    except RuntimeError:
        print("DataParallel weight detected. loading...")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_for_loading.items():
            name = k.replace("module.", "")  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, scheduler, iteration


def load_checkpoint_warm_start(checkpoint_path, model, optimizer, scheduler):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']

    model_for_loading = checkpoint_dict['model']

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_for_loading.items() if
                       (k in model_dict) and (model_dict[k].shape == model_for_loading[k].shape)}
    model_dict.update(pretrained_dict)
    missing_and_unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    print(
        "WARNING: only part of the model loaded. below are missing and unexpected keys, make sure that they are correct:")
    print(missing_and_unexpected_keys)

    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, scheduler, iteration


def load_averaged_checkpoint_warm_start(checkpoint_path, model, optimizer, scheduler):
    # checkpoint_path is dir in this function
    assert os.path.isdir(checkpoint_path)

    list_checkpoints = last_n_checkpoints(checkpoint_path, args.average_checkpoint)
    iteration = 0

    model_for_loading = average_checkpoints(list_checkpoints, args.epsilon)['model']

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_for_loading.items() if
                       (k in model_dict) and (model_dict[k].shape == model_for_loading[k].shape)}
    model_dict.update(pretrained_dict)
    missing_and_unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    print(
        "WARNING: only part of the model loaded. below are missing and unexpected keys, make sure that they are correct:")
    print(missing_and_unexpected_keys)

    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, scheduler, iteration


def save_checkpoint(model, optimizer, scheduler, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()  # dataparallel case
    else:
        model_state_dict = model.state_dict()

    torch.save({'model': model_state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(model, num_gpus, output_directory, epochs, learning_rate, lr_decay_step, lr_decay_gamma,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    # local eval and synth functions
    model.train()
    def evaluate():
        # eval loop
        model.eval()
        epoch_eval_loss = 0
        for i, batch in enumerate(test_loader):
            with torch.no_grad():
                mel, audio = batch
                mel = torch.autograd.Variable(mel.cuda())
                audio = torch.autograd.Variable(audio.cuda())
                outputs = model(audio, mel)

                loss = criterion(outputs)
                if num_gpus > 1:
                    reduced_loss = loss.mean().item()
                else:
                    reduced_loss = loss.item()
                epoch_eval_loss += reduced_loss

        epoch_eval_loss = epoch_eval_loss / len(test_loader)
        print("EVAL {}:\t{:.9f}".format(iteration, epoch_eval_loss))
        if with_tensorboard:
            logger.add_scalar('eval_loss', epoch_eval_loss, iteration)
            logger.flush()
        model.train()

    def synthesize(sigma):
        model.eval()
        # synthesize loop
        # model.h_cache = model.module.cache_flow_embed()

        for i, batch in enumerate(synth_loader):
            if i == 0:
                with torch.no_grad():
                    mel, _, filename = batch
                    mel = torch.autograd.Variable(mel.cuda())
                    try:
                        model.h_cache = model.cache_flow_embed()
                        audio = model.reverse(mel, sigma)
                    except AttributeError:
                        model.module.h_cache = model.module.cache_flow_embed()
                        audio = model.module.reverse(mel, sigma)
                    except NotImplementedError:
                        print("reverse not implemented for this model. skipping synthesize!")
                        model.train()
                        return

                    audio = audio * MAX_WAV_VALUE
                audio = audio.squeeze()
                audio = audio.cpu().numpy()
                audio = audio.astype('int16')
                audio_path = os.path.join(
                    os.path.join(output_directory, "samples", waveflow_config["model_name"]),
                    "generate_{}.wav".format(iteration))
                write(audio_path, data_config["sampling_rate"], audio)

        model.train()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    criterion = WaveFlowLossDataParallel(sigma)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    model.h_cache = model.cache_flow_embed()

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        if args.warm_start:
            print("INFO: --warm_start. optimizer and scheduler are initialized and strict=False for load_state_dict().")
            if args.average_checkpoint == 0:
                model, optimizer, scheduler, iteration = load_checkpoint_warm_start(checkpoint_path, model, optimizer,
                                                                                    scheduler)
            else:
                print("INFO: --average_checkpoint > 0. loading an averaged weight of last {} checkpoints...".format(
                    args.average_checkpoint))
                model, optimizer, scheduler, iteration = load_averaged_checkpoint_warm_start(checkpoint_path, model,
                                                                                             optimizer, scheduler)
        else:
            model, _, scheduler, iteration = load_checkpoint(checkpoint_path, model,
                                                                     optimizer, scheduler)
        iteration += 1  # next iteration is iteration + 1

    if num_gpus > 1:
        print("num_gpus > 1. converting the model to DataParallel...")
        model = torch.nn.DataParallel(model)

    trainset = Mel2Samp("train", False, False, **data_config)
    train_loader = DataLoader(trainset, num_workers=4, shuffle=True,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    testset = Mel2Samp("test", False, False, **data_config)
    test_sampler = None
    test_loader = DataLoader(testset, num_workers=4, shuffle=False,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)

    synthset = Mel2Samp("test", True, True, **data_config)
    synth_sampler = None
    synth_loader = DataLoader(synthset, num_workers=4, shuffle=False,
                              sampler=synth_sampler,
                              batch_size=1,
                              pin_memory=False,
                              drop_last=False)

    # Get shared output_directory ready
    if not os.path.isdir(os.path.join(output_directory, waveflow_config["model_name"])):
        os.makedirs(os.path.join(output_directory, waveflow_config["model_name"]), exist_ok=True)
        os.chmod(os.path.join(output_directory, waveflow_config["model_name"]), 0o775)
    print("output directory", os.path.join(output_directory, waveflow_config["model_name"]))
    if not os.path.isdir(os.path.join(output_directory, "samples")):
        os.makedirs(os.path.join(output_directory, "samples"), exist_ok=True)
        os.chmod(os.path.join(output_directory, "samples"), 0o775)
    os.makedirs(os.path.join(output_directory, "samples", waveflow_config["model_name"]), exist_ok=True)
    os.chmod(os.path.join(output_directory, "samples", waveflow_config["model_name"]), 0o775)

    if with_tensorboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, waveflow_config["model_name"], 'logs'))

    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=0.1, last_epoch)
    
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            tic = time.time()

            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model(audio, mel)
            # print("loss logdet: ", outputs[1])

            loss = criterion(outputs)
            # loss = loss*scale_loss
            if torch.isnan(loss):
                print("!!! Loss is NaN")
                continue
            # if scale_loss is not None:
            #     loss = loss*scale_loss
            if num_gpus > 1:
                reduced_loss = loss.mean().item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.mean().backward()

            if fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            toc = time.time() - tic

            print("{}:\t{:.9f}, {:.4f} seconds".format(iteration, reduced_loss, toc))
            if with_tensorboard:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)
                logger.add_scalar('lr', get_lr(optimizer), i + len(train_loader) * epoch)
                logger.add_scalar('grad_norm', grad_norm, i + len(train_loader) * epoch)
                logger.flush()

            if (iteration % iters_per_checkpoint == 0):
                checkpoint_path = "{}/flowvocoder_{}".format(
                    os.path.join(output_directory, waveflow_config["model_name"]), iteration)
                save_checkpoint(model, optimizer, scheduler, learning_rate, iteration,
                                checkpoint_path)

                if iteration != 0:
                    evaluate()
                    del mel, audio, outputs, loss
                    gc.collect()
                    synthesize(sigma)

            iteration += 1
            scheduler.step()

        evaluate()


def synthesize_master(model, num_gpus, temp, output_directory, epochs, learning_rate, lr_decay_step, lr_decay_gamma,
                      sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
                      checkpoint_path, with_tensorboard):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model.h_cache = model.cache_flow_embed()
    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, _, _, iteration = load_checkpoint(checkpoint_path, model, None, None)

    if hasattr(model, 'cache_flow_embed'):
        model.h_cache = model.cache_flow_embed(remove_after_cache=True)
    # remove all weight_norm from the model
    model.remove_weight_norm()
    model.fuse_conditioning_layers()

    # fuse mel-spec conditioning layer weights to maximize speed
    

    if fp16_run:
        from apex import amp
        model, _ = amp.initialize(model, [], opt_level="O3")

    synthset = Mel2Samp("test", True, True, **data_config)
    synth_sampler = None
    synth_loader = DataLoader(synthset, num_workers=4, shuffle=False,
                              sampler=synth_sampler,
                              batch_size=1,
                              pin_memory=False,
                              drop_last=False)

    # Get shared output_directory ready
    if not os.path.isdir(os.path.join(output_directory, waveflow_config["model_name"])):
        os.makedirs(os.path.join(output_directory, waveflow_config["model_name"]), exist_ok=True)
        os.chmod(os.path.join(output_directory, waveflow_config["model_name"]), 0o775)
    print("output directory", os.path.join(output_directory, waveflow_config["model_name"]))
    if not os.path.isdir(os.path.join(output_directory, "samples")):
        os.makedirs(os.path.join(output_directory, "samples"), exist_ok=True)
        os.chmod(os.path.join(output_directory, "samples"), 0o775)
    os.makedirs(os.path.join(output_directory, "samples", waveflow_config["model_name"]), exist_ok=True)
    os.chmod(os.path.join(output_directory, "samples", waveflow_config["model_name"]), 0o775)

    # synthesize loop
    model.eval()
    for i, batch in enumerate(synth_loader):
        with torch.no_grad():
            mel, _, filename = batch
            mel = torch.autograd.Variable(mel.cuda())
            if fp16_run:
                mel = mel.half()

            torch.cuda.synchronize()
            tic = time.time()
            audio = model.reverse_fast(mel, temp)
            torch.cuda.synchronize()
            toc = time.time() - tic
            print("EST time: ", toc)

            print('{}: {:.4f} seconds, {:.4f}kHz'.format(i, toc, audio.shape[1] / toc / 1000))

        audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            os.path.join(output_directory, "samples", waveflow_config["model_name"]),
            "generate_{}_{}_t{}.wav".format(iteration, i, temp))

        write(audio_path, data_config["sampling_rate"], audio)

    model.train()

def synthesize_tacotron2(model, num_gpus, temp, gen_mel_dir, output_directory, epochs, learning_rate, lr_decay_step, lr_decay_gamma,
                      sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
                      checkpoint_path, with_tensorboard):
    import glob
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load checkpoint if one exists
    iteration = 0
    model.h_cache = model.cache_flow_embed()
    if checkpoint_path != "":
        model, _, _, iteration = load_checkpoint(checkpoint_path, model, None, None)

    if hasattr(model, 'cache_flow_embed'):
        model.h_cache = model.cache_flow_embed(remove_after_cache=True)  # used for flow conditioning models

    list_mel = glob.glob(os.path.join(gen_mel_dir,"*.npy"))
    list_mel = [(os.path.basename(mel), np.load(mel)) for mel in list_mel]

    # Get shared output_directory ready
    if not os.path.isdir(os.path.join(output_directory, waveflow_config["model_name"])):
        os.makedirs(os.path.join(output_directory, waveflow_config["model_name"]), exist_ok=True)
        os.chmod(os.path.join(output_directory, waveflow_config["model_name"]), 0o775)
    print("output directory", os.path.join(output_directory, waveflow_config["model_name"]))
    if not os.path.isdir(os.path.join(output_directory, "samples")):
        os.makedirs(os.path.join(output_directory, "samples"), exist_ok=True)
        os.chmod(os.path.join(output_directory, "samples"), 0o775)
    os.makedirs(os.path.join(output_directory, "samples", waveflow_config["model_name"]), exist_ok=True)
    os.chmod(os.path.join(output_directory, "samples", waveflow_config["model_name"]), 0o775)

    # synthesize loop
    model.eval()

    for batch in list_mel:
        with torch.no_grad():
            filename, mel = batch[0], batch[1]
            mel = torch.autograd.Variable(torch.from_numpy(mel).float().cuda())
            mel = mel.unsqueeze(0)
            if fp16_run:
                mel = mel.half()
            torch.cuda.synchronize()
            tic = time.time()
            audio = model.reverse(mel, temp)
            torch.cuda.synchronize()
            toc = time.time() - tic
            print('{:.4f} seconds, {:.4f}kHz'.format(toc, audio.shape[1] / toc / 1000))
        audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(gen_mel_dir, "samples", filename.replace("npy","wav"))
        write(audio_path, data_config["sampling_rate"], audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-w', '--warm_start', action='store_true',
                        help='warm start. i.e. load_state_dict() with strict=False and optimizer & scheduler are initialized.')
    parser.add_argument('-s', '--synthesize', action='store_true',
                        help='run synthesize loop only. does not train or evaluate the model.')
    parser.add_argument('-tr', '--train', action='store_true',
                        help='training option')
    parser.add_argument('--synthesize_tacotron2', action='store_true',
                        help='generate audio from synthetic mel spectrogram from text using pretrained Tacotron2 model')
    parser.add_argument('-t', '--temp', type=float, default=1.,
                        help='temperature during synthesize loop. defaults to 1. only applicable if -s is specified')
    parser.add_argument('-a', '--average_checkpoint', type=int, default=0,
                        help='checkpoint averaging. averages the given number of latest checkpoints for synthesize.')
    parser.add_argument('-e', '--epsilon', type=float, default=None,
                        help='epsilon value for polyak averaging. only applied if -a > 0. defaults to None (plain averaging)')
    parser.add_argument('--gen_mel_dir', type=str, default="",
                        help='directory of mel spectrogram from pretrained tacotron2 model')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global waveflow_config
    waveflow_config = config["model_config"]

    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = build_model(waveflow_config)
    

    if args.synthesize:
        print("INFO: --synthesize is true. running only synthesize loop...")
        synthesize_master(model, num_gpus, args.temp, **train_config)
        print("INFO: synthesize loop done. exiting!")
        exit()
    elif args.train:
        train(model, num_gpus, **train_config)
    elif args.synthesize_tacotron:
        synthesize_tacotron2(model, num_gpus, args.temp, args.gen_mel_dir, **train_config)