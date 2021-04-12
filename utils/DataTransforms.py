from torch import nn
import torchaudio
import torch
import albumentations as A
import numpy as np
from TextTransforms import TextTransform

class DataCollate:
  def __init__(self, sr=16000, n_mels=128, audio_aug=None, specAug=None, specCut:bool=None, holes:int=24):
    self.sr = sr
    self.transform = torchaudio.transforms.MelSpectrogram(
      n_fft=1024,
      sample_rate=self.sr,
      n_mels=n_mels,
      win_length=1024,
      hop_length=256,
    )
    self.spec_aug = nn.Sequential(
      torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
      torchaudio.transforms.TimeMasking(time_mask_param=50)
    ) if specAug else None
    self.cutout = specCut
    self.text_transform = TextTransform()
    self.audio_aug=audio_aug
    self.holes = holes


  def __call__(self, ls_batch):
    spectrograms = []
    input_lengths = []
    targets = []
    target_lengths = []

    for waveform, _, utterance, _, _, _ in ls_batch:
      if self.audio_aug:
        waveform, self.sr = torchaudio.sox_effects.apply_effects_tensor(waveform, self.sr, self.audio_aug, channels_first=True)
        self.transform.sample_rate = self.sr
      
      mspec = torch.log(self.transform(waveform).clamp(1e-5)).squeeze(0)
      if self.cutout:
        np_spec = A.CoarseDropout(max_holes=self.holes, min_height=4, max_height=16, min_width=4, max_width=16, p=0.9)(image=mspec.cpu().numpy())['image']
        mspec = torch.from_numpy(np_spec)
      mspec = mspec.transpose(0,1)

      if self.spec_aug:
        mspec = self.spec_aug(mspec)

      spectrograms.append(mspec)
      input_lengths.append(mspec.shape[0] // 2)
      targets.append(torch.Tensor(self.text_transform.text_to_int(utterance)))
      target_lengths.append(len(targets[-1]))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).transpose(1,2)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return spectrograms, targets, input_lengths, target_lengths