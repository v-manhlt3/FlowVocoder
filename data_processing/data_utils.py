import random
import numpy as np 
import torch
import torch.utils.data as data

import commons
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict
from text.symbols import symbols


class TextMelLoader(data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False)
        if getattr(hparams, "cmudict_path", None) is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        self.stft = commons.TacotronsSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
    
    def get_mel_text_pair(self, audiopaths_and_text):
        audiopath, text = audiopaths_and_text[0], audiopaths_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR does not match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels)
            )
        return melspec
    
    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols))
        text_norm = torch.IntTensor(text_norm)
        return text_norm
    
    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])
    
    def __len__(self):
        return len(self.audiopaths_and_text)

class TextMelCollate():
    """
        Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step
    
    def __cal__(self, batch):
        """
        Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        -------------------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all onehot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, output_lengths

""" Multi-speaker Version"""
# class TextMelSpeakerLoader(data.Dataset):
#     """
#         1) loads audio, speaker_id, text pairs
#         2) normalizes text and converts them to sequences of one-hot vectors
#         3) computes mel-spectrograms from audio files.
#     """
#     def __init__(self, audiopaths_sid_text, hparams):
#         self.audiopaths_and_text = load_filepaths_and_text(audiopaths_sid_text)