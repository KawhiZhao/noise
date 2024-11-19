# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
import librosa
from torchvision import transforms
import pdb
import random
import sys
sys.path.append(op.abspath(op.join(op.dirname(__file__), '..')))
from simulate.utils_simulate_speech import mix_single_audio
from simulate.main_simulate import cal_rms, cal_adjusted_rms
# from tools.hparams import *
# from tools.audio import Audio

class SimulatedSpeechStft(data.Dataset):
    def __init__(self, phrase='train', audio_embedding=True, snr=-1):

        self.phrase = phrase

        self.sample_rate = 16000
        self.durations = 3

        with open(os.path.join('speech_dataset', phrase + '.txt'), 'r') as f:
            contents = f.readlines()
            audio_list = contents
        
        # get the list of audio files whose duration is less than 3s
        with open(os.path.join('speech_dataset', phrase + '_filtered.txt'), 'r') as f:
            contents = f.readlines()
            audio_list_filtered = contents
        
        # remove the audio files whose duration is less than 3s from the audio_list
        for line in audio_list_filtered:
            audio_list.remove(line)
        
        self.audio_list = audio_list

        audio_set_dir = '/mnt/fast/datasets/audio/audio_classification_datasets/esc50/ESC-50-master/audio/'

        audio_set_list = os.listdir(audio_set_dir)
        self.noise_list = [audio_set_dir + audio for audio in audio_set_list]

        self.snr = snr
        
        
    def __len__(self):
        if self.phrase == 'train':
            return 50000
        else:
            return 1000
  


    def __getitem__(self, idx):

        
        mixed_audio, doa = mix_single_audio(1, self.phrase, self.audio_list)

        start_point = np.random.randint(0, mixed_audio.shape[1] - self.sample_rate * self.durations)
        mixed_audio = mixed_audio[:, start_point:start_point + self.sample_rate * self.durations]

        if self.snr != -1:
            random_len = 0
            while random_len < 3:
                
                random_nb = np.random.randint(low=0, high=len(self.noise_list))
                noise_path = self.noise_list[random_nb].strip()
                random_len = librosa.get_duration(filename=noise_path, sr=48000)


            divided_noise_amp, sr = librosa.load(noise_path, sr=48000)

            random_start = np.random.randint(0, divided_noise_amp.shape[0] - 48000 * self.durations)
            divided_noise_amp = divided_noise_amp[random_start:random_start + 48000 * self.durations]

            divided_noise_amp = librosa.resample(divided_noise_amp, 48000, 16000)

            # create audio mixture with a specific SNR level
            source_power = np.mean(mixed_audio ** 2)
            noise_power = np.mean(divided_noise_amp ** 2)

            desired_noise_power = source_power / (10 ** (self.snr / 10))
            scaling_factor = np.sqrt(desired_noise_power / noise_power)
            noise_waveform = divided_noise_amp * scaling_factor
            # extend the noise to multi-channel
            noise_waveform = np.tile(noise_waveform, (mixed_audio.shape[0], 1))

            mixed_waveform = mixed_audio + noise_waveform

            max_value = np.max(np.abs(mixed_waveform))  #  # normalize the mixture need to be done
            if max_value > 1:
                mixed_waveform *= 0.9 / max_value
            
            mixed_audio = mixed_waveform
            


        # extract STFT for multi-channel audio
        reals = []
        imags = []
        for i in range(mixed_audio.shape[0]):
            try:
                stft = librosa.stft(mixed_audio[i], n_fft=512, hop_length=160)
            except:
                pdb.set_trace()
            real = stft.real
            imag = stft.imag
            reals.append(real)
            imags.append(imag)
        reals = reals + imags
        # stack the real and imag part of the STFT
        mixed_mag_phase = np.stack(reals, axis=0)
        return mixed_mag_phase, doa
     
        

        
        
        

if __name__ == '__main__':
    simulated_data = SimulatedSpeechSTFT()
    dataloader = data.DataLoader(simulated_data, batch_size=2, shuffle=False, num_workers=0)
    for i, (mixed_mag_phase, doa) in enumerate(dataloader):
        pdb.set_trace()
        if i == 1:
            break
    