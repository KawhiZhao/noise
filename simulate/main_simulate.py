
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import argparse
import librosa
import soundfile as sf
from datetime import datetime
import os
from tqdm import tqdm
import random
import csv
import math
# min_z = 5
# max_z = 15
# min_x = 5
# max_x = 15
# min_y = 5
# max_y = 15
min_z = 3
max_z = 3
min_x = 3
max_x = 3
min_y = 3
max_y = 3
methods = ["ism", "hybrid"]

# min_rt60 = 0.5
# max_rt60 = 1

min_rt60 = 0.1
max_rt60 = 0.1

# min_speaker_z = 1.5
# max_speaker_z = 1.9

# min_mic_z = 1
# max_mic_z = 1.2

# min_mic_dis = 0.1
# max_mic_dis = 0.13

min_speaker_z = 1.6
max_speaker_z = 1.6

min_mic_z = 1
max_mic_z = 1

min_mic_dis = 0.1
max_mic_dis = 0.1

snr_pool = [-10, 0, 10, 20]

def calculate_distance(speaker_x, speaker_y, mic_x, mic_y):
    delta_x = speaker_x - mic_x
    delta_y = speaker_y - mic_y

    distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    return distance

def calculate_DoA(speaker_x, speaker_y, mic_x, mic_y):
    delta_x = speaker_x - mic_x
    delta_y = speaker_y - mic_y
    # normalize delta_x and delta_y to make sure the distance is 1
    # distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    # delta_x_norm = delta_x / distance
    # delta_y_norm = delta_y / distance
    doa = np.arctan2(delta_y, delta_x) * 180 / np.pi
    if doa < 0:
        doa = doa + 360
    return doa

def calculate_DoA_norm(speaker_x, speaker_y, mic_x, mic_y):
    delta_x = speaker_x - mic_x
    delta_y = speaker_y - mic_y
    # normalize delta_x and delta_y to make sure the distance is 1
    distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    delta_x_norm = delta_x / distance
    delta_y_norm = delta_y / distance
    doa = np.arctan2(delta_y, delta_x) * 180 / np.pi
    if doa < 0:
        doa = doa + 360
    return doa, delta_x_norm, delta_y_norm

def judge(position_list, mic_position):
    nb_sources = len(position_list)
    mic_x, mic_y, mic_z = mic_position
    doa_list = []
    for i in range(nb_sources):
        speaker_x, speaker_y, speaker_z = position_list[i]
        delta_x = speaker_x - mic_x
        delta_y = speaker_y - mic_y

        doa = np.arctan2(delta_y, delta_x) * 180 / np.pi
        if doa < 0:
            doa = doa + 360

        doa_list.append(doa)
    # import pdb;pdb.set_trace()
    for i in range(nb_sources - 1):
        for j in range(i + 1, nb_sources):
            if (360 - np.abs(doa_list[i] - doa_list[j])) < 90:
                return False
    
    return True



def get_random_snr():
    random_nb = np.random.randint(low=0, high=len(snr_pool))
    return snr_pool[random_nb]

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def get_identifier():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

def get_random_number(low, high):

    return (high - low) * np.random.random() + low

def get_room_size():
    
    x = (max_x - min_x) * np.random.random() + min_x
    y = (max_y - min_y) * np.random.random() + min_y
    z = (max_z - min_z) * np.random.random() + min_z
    return x, y, z

def get_rt60():
    
    rt60 = (max_rt60 - min_rt60) * np.random.random() + min_rt60
    return rt60

def get_source_position(room_x, room_y):

    min_dis = 0.1
    max_x = room_x - min_dis
    max_y = room_y - min_dis
    x = (max_x - min_dis) * np.random.random() + min_dis
    y = (max_y - min_dis) * np.random.random() + min_dis
    z = (max_speaker_z - min_speaker_z) * np.random.random() + min_speaker_z
    return x, y, z

def get_source_position_fix_r(center_x, center_y, r):
    
    theta = random.uniform(0, 2 * math.pi)  # Random angle in radians
    x1 = center_x + r * math.cos(theta)
    y1 = center_y + r * math.sin(theta)
    
    z = (max_speaker_z - min_speaker_z) * np.random.random() + min_speaker_z
    return x1, y1, z

def get_mic_position(room_x, room_y):
    z = get_random_number(min_mic_z, max_mic_z)
    mic_dis = get_random_number(min_mic_dis, max_mic_dis)
    dis = 2 * mic_dis
    center_x = get_random_number(dis, room_x - dis)
    center_y = get_random_number(dis, room_y - dis)
    pos_up = [center_x, center_y + mic_dis, z]
    pos_down = [center_x, center_y - mic_dis, z]
    pos_left = [center_x - mic_dis, center_y, z]
    pos_right = [center_x + mic_dis, center_y, z]
    pos = np.c_[
        pos_up, pos_down, pos_left, pos_right
    ]
    
    return pos, center_x, center_y, mic_dis, z

def get_mic_position_fix (room_x, room_y):
    z = get_random_number(min_mic_z, max_mic_z)
    mic_dis = get_random_number(min_mic_dis, max_mic_dis)
    dis = 2 * mic_dis
    center_x = room_x / 2
    center_y = room_y / 2
    pos_up = [center_x, center_y + mic_dis, z]
    pos_down = [center_x, center_y - mic_dis, z]
    pos_left = [center_x - mic_dis, center_y, z]
    pos_right = [center_x + mic_dis, center_y, z]
    pos = np.c_[
        pos_up, pos_down, pos_left, pos_right
    ]
    
    return pos, center_x, center_y, mic_dis, z

def get_audio_list():
    with open('LibriSpeech-test/audio_list_test.txt', 'r') as f:
        contents = f.readlines()
    return contents

def get_AAC_list(phrase):
    csv_path = '/mnt/fast/nobackup/scratch4weeks/jz01019/DCASE2022-data-generator/audiocaps/dataset/***.csv'
    csv_path = csv_path.replace('***', phrase)
    captions_dict = {}
    with open(csv_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # find if row['caption'] contains double quotes, if so, convert to single quotes
            if '"' in row['caption']:
                tmp = row['caption'].replace('"', "'")
                captions_dict[row['youtube_id']] = tmp
            else:
                captions_dict[row['youtube_id']] = row['caption']
    
    return captions_dict

def get_noise_list():
    with open('./noise_files.txt', 'r') as f:
        contents = f.readlines()
    return contents

def mix_single_audio(count):
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[1],
        help="Simulation method to use",
    )
    args = parser.parse_args()

    rt60_tgt = get_rt60()

    room_x, room_y, room_z = get_room_size()
    room_dim = [room_x, room_y, room_z]

    audio_list = get_audio_list()
    # audio_list = audio_list[start:end]
    # noise_list = get_noise_list()
    # find /Volumes/jz01019/roomacoustic/CMU_ARCTIC  -maxdepth 3 -type f -iname "*.wav"  > audio_list.txt
    # fs, audio1 = wavfile.read(audio_list[0].strip())
    for i in range(count):
    # for audio_p in tqdm(audio_list):
        audio_index = int(get_random_number(0, len(audio_list)))
        audio_p = audio_list[audio_index].strip()
        # audio_p = audio_p.strip()
        # fs, audio1 = wavfile.read(audio_p)
        audio1, fs = librosa.load(audio_p, sr=16000)
        len_audio = len(audio1)

        # random_nb = np.random.randint(low=0, high=len(noise_list))
        # noise_path = noise_list[random_nb].strip()

        # noise_audio, fs = librosa.load(noise_path, sr=16000)
        # len_noise = len(noise_audio)

        # start = random.randint(0, len_noise - len_audio)
        # divided_noise_amp = noise_audio[start : start + len_audio]
        # noise_rms = cal_rms(divided_noise_amp)
        # clean_rms = cal_rms(audio1)
        # snr = get_random_snr()
        # adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        # adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / (noise_rms))

        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

        # Create the room
        if args.method == "ism":
            room = pra.ShoeBox(
                room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
            )
        elif args.method == "hybrid":
            room = pra.ShoeBox(
                room_dim,
                fs=fs,
                materials=pra.Material(e_absorption),
                max_order=3,
                ray_tracing=True,
                air_absorption=True,
            )
        speaker_x, speaker_y, speaker_z = get_source_position(room_x, room_y)
        # noise_x, noise_y, noise_z = get_source_position(room_x, room_y)

        room.add_source([speaker_x, speaker_y, speaker_z], signal=audio1)
        # room.add_source([noise_x, noise_y, noise_z], signal=adjusted_noise_amp)
        

        mic_locs, center_x, center_y, mic_dis, mic_z = get_mic_position(room_x, room_y)

        room.add_microphone_array(mic_locs)

        room.simulate()

        room_str = str(len_audio)
        folder_name = get_identifier() + '_' + room_str
        wav_name = folder_name + '.wav'
        txt_name = folder_name + '.txt'
        folder_name = os.path.join('/mnt/fast/nobackup/scratch4weeks/jz01019/tmp1/test_dataset', 'onespeaker', folder_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        room.mic_array.to_wav(
            f"./tmp/" + wav_name,
            norm=True,
            bitdepth=np.int16,
        )

        y, sr = librosa.load("./tmp/" + wav_name, sr=fs, mono=False)
        # import pdb; pdb.set_trace()
        y = y[:, :len_audio]
        y = y.transpose()
        
        sf.write(os.path.join(folder_name, wav_name), y, samplerate=fs)
        

        with open(os.path.join(folder_name, txt_name), 'w') as f:
            f.write('speaker no: 1\n')
            f.write('speaker 1 position: ' + str(speaker_x) + ' ' + str(speaker_y) + ' ' + str(speaker_z) + '\n')
            f.write('mic position: ' + str(center_x) + ' ' + str(center_y) + ' ' + str(mic_dis) + ' ' + str(mic_z) + '\n')
            f.write('rt60: ' + str(rt60_tgt) + '\n')
            f.write('room size: ' + str(room_x) + ' ' + str(room_y) + ' ' + str(room_z) + '\n')
            # f.write('noise position: ' + str(noise_x) + ' ' + str(noise_y) + ' ' + str(noise_z) + '\n')
            # f.write('snr: ' + str(snr) + '\n')


if __name__ == '__main__':
    mix_single_audio(1)