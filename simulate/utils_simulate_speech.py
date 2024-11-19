from simulate.main_simulate import *
# from main_simulate import *
import pdb
from tqdm import tqdm

def mix_single_audio(count, phrase='train', audio_list=None):
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

    
        

    # noise_list = get_noise_list()
    # find /Volumes/jz01019/roomacoustic/CMU_ARCTIC  -maxdepth 3 -type f -iname "*.wav"  > audio_list.txt
    # fs, audio1 = wavfile.read(audio_list[0].strip())
    for i in range(count):
        audio_path_1 = audio_list[int(get_random_number(0, len(audio_list)))]
        
        audio_path_1 = audio_path_1.strip()
        audio_path_1 = audio_path_1.replace('/mnt/fast/nobackup/users/jz01019/', '/mnt/fast/nobackup/scratch4weeks/jz01019/')
        audio1, fs = librosa.load(audio_path_1, sr=16000)
        len_audio_1 = len(audio1)
        

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
        mic_locs, center_x, center_y, mic_dis, mic_z = get_mic_position_fix(room_x, room_y)

        speaker_x, speaker_y, speaker_z = get_source_position_fix_r(center_x, center_y, 0.8)
        
        
        # noise_x, noise_y, noise_z = get_source_position(room_x, room_y)

        room.add_source([speaker_x, speaker_y, speaker_z], signal=audio1)
        
        # room.add_source([noise_x, noise_y, noise_z], signal=adjusted_noise_amp)
        

        

        room.add_microphone_array(mic_locs)

        room.simulate()

        
        doa1 = calculate_DoA(speaker_x, speaker_y, center_x, center_y)
        
        doa1 = int(doa1)
        distance1 = calculate_distance(speaker_x, speaker_y, center_x, center_y)
        # assert distance1.float() == 0.8, distance1
        folder_name = get_identifier()
        wav_name = folder_name + '.wav'

        mics_signals = room.mic_array.signals
        mics_signals = mics_signals[:, :len_audio_1]
        
    
        return mics_signals, doa1

def mix_two_audio(count, phrase='train'):
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

    with open(os.path.join('/mnt/fast/nobackup/scratch4weeks/jz01019/AAC-SELD/speech_dataset', phrase + '.txt'), 'r') as f:
        contents = f.readlines()
        audio_list = contents
    
    # get the list of audio files whose duration is less than 3s
    with open(os.path.join('/mnt/fast/nobackup/scratch4weeks/jz01019/AAC-SELD/speech_dataset', phrase + '_filtered.txt'), 'r') as f:
        contents = f.readlines()
        audio_list_filtered = contents
    
    # remove the audio files whose duration is less than 3s from the audio_list
    for line in audio_list_filtered:
        audio_list.remove(line)
        

    # noise_list = get_noise_list()
    # find /Volumes/jz01019/roomacoustic/CMU_ARCTIC  -maxdepth 3 -type f -iname "*.wav"  > audio_list.txt
    # fs, audio1 = wavfile.read(audio_list[0].strip())
    for i in range(count):
        while(True):
            audio_path_1 = audio_list[int(get_random_number(0, len(audio_list)))]
            audio_path_2 = audio_list[int(get_random_number(0, len(audio_list)))]
            speaker_id_1 = audio_path_1.split('/')[-3]
            speaker_id_2 = audio_path_2.split('/')[-3]
            if speaker_id_1 != speaker_id_2:
                break
        audio_path_1 = audio_path_1.strip()
        audio_path_2 = audio_path_2.strip()
        audio1, fs = librosa.load(audio_path_1, sr=16000)
        audio2, fs = librosa.load(audio_path_2, sr=16000)
        len_audio_1 = len(audio1)
        len_audio_2 = len(audio2)
        len_audio = min(len_audio_1, len_audio_2)
        audio1 = audio1[:len_audio]
        audio2 = audio2[:len_audio]

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
        mic_locs, center_x, center_y, mic_dis, mic_z = get_mic_position(room_x, room_y)

        speaker_x, speaker_y, speaker_z = get_source_position(room_x, room_y)
        speaker_x2, speaker_y2, speaker_z2 = get_source_position(room_x, room_y)
        while (not judge([(speaker_x, speaker_y, speaker_z), (speaker_x2, speaker_y2, speaker_z2)], (center_x, center_y, mic_z))):
            speaker_x, speaker_y, speaker_z = get_source_position(room_x, room_y)
            speaker_x2, speaker_y2, speaker_z2 = get_source_position(room_x, room_y)
        # noise_x, noise_y, noise_z = get_source_position(room_x, room_y)

        room.add_source([speaker_x, speaker_y, speaker_z], signal=audio1)
        room.add_source([speaker_x2, speaker_y2, speaker_z2], signal=audio2)
        # room.add_source([noise_x, noise_y, noise_z], signal=adjusted_noise_amp)
        

        

        room.add_microphone_array(mic_locs)

        room.simulate()

        room_str = str(len_audio)
        doa1 = calculate_DoA(speaker_x, speaker_y, center_x, center_y)
        doa2 = calculate_DoA(speaker_x2, speaker_y2, center_x, center_y)
        doa1 = int(doa1)
        distance1 = calculate_distance(speaker_x, speaker_y, center_x, center_y)
        distance2 = calculate_distance(speaker_x2, speaker_y2, center_x, center_y)
        folder_name = get_identifier()
        wav_name = folder_name + '.wav'
        txt_name = folder_name + '.txt'
        # folder_name = os.path.join('/mnt/fast/nobackup/scratch4weeks/jz01019/AAC/tmp/', folder_name)
        # if not os.path.exists(folder_name):
        #     os.mkdir(folder_name)
        output_path = os.path.join("/mnt/fast/nobackup/scratch4weeks/jz01019/AAC/tmp_speech/", wav_name)
        mics_signals = room.mic_array.signals
        mics_signals = mics_signals[:, :len_audio]
        # pdb.set_trace()
        # room.mic_array.to_wav(
        #     output_path,
        #     norm=True,
        #     bitdepth=np.int16,
        # )

        # y, sr = librosa.load("/mnt/fast/nobackup/scratch4weeks/jz01019/AAC/tmp/" + wav_name, sr=fs, mono=False)
        # y = y[:, :len_audio]
        # # padding the audio to 11 seconds if the audio is shorter than 11 seconds
        # if len_audio < 11 * fs:
        #     y = np.pad(y, ((0, 11 * fs - len_audio), (0, 0)), 'constant', constant_values=0)
        # import pdb; pdb.set_trace()
        # y = y[:, :len_audio]
        # y = y.transpose()
        
        # sf.write(os.path.join(folder_name, wav_name), y, samplerate=fs)
        # calculate DoA given the position of the speaker and the microphone (center_x, center_y, mic_z)
        

        # with open(os.path.join(folder_name, txt_name), 'w') as f:
        #     f.write('speaker no: 2\n')
        #     f.write('speaker 1 position: ' + str(speaker_x) + ' ' + str(speaker_y) + ' ' + str(speaker_z) + '\n')
        #     f.write('speaker 2 position: ' + str(speaker_x2) + ' ' + str(speaker_y2) + ' ' + str(speaker_z2) + '\n')
        #     f.write('mic position: ' + str(center_x) + ' ' + str(center_y) + ' ' + str(mic_dis) + ' ' + str(mic_z) + '\n')
        #     f.write('rt60: ' + str(rt60_tgt) + '\n')
        #     f.write('room size: ' + str(room_x) + ' ' + str(room_y) + ' ' + str(room_z) + '\n')
    
        return mics_signals, audio1, doa1, len_audio, distance1 < distance2

def mix_three_audio(count):
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
 
    # noise_list = get_noise_list()
    # find /Volumes/jz01019/roomacoustic/CMU_ARCTIC  -maxdepth 3 -type f -iname "*.wav"  > audio_list.txt
    # fs, audio1 = wavfile.read(audio_list[0].strip())
    for i in range(count):
        while(True):
            audio_1_index = int(get_random_number(0, len(audio_list)))
            audio_2_index = int(get_random_number(0, len(audio_list)))
            audio_3_index = int(get_random_number(0, len(audio_list)))
            if (audio_1_index != audio_2_index) and (audio_1_index != audio_3_index) and (audio_2_index != audio_3_index):
                break
        audio_p1 = audio_list[audio_1_index].strip()
        audio_p2 = audio_list[audio_2_index].strip()
        audio_p3 = audio_list[audio_3_index].strip()
        # audio_p = audio_p.strip()
        # fs, audio1 = wavfile.read(audio_p)
        audio1, fs = librosa.load(audio_p1, sr=16000)
        audio2, fs = librosa.load(audio_p2, sr=16000)
        audio3, fs = librosa.load(audio_p3, sr=16000)
        len_audio_1 = len(audio1)
        len_audio_2 = len(audio2)
        len_audio_3 = len(audio3)

        len_audio = min(len_audio_1, len_audio_2, len_audio_3)
        audio1 = audio1[:len_audio]
        audio2 = audio2[:len_audio]
        audio3 = audio3[:len_audio]

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
        mic_locs, center_x, center_y, mic_dis, mic_z = get_mic_position(room_x, room_y)
        speaker_x, speaker_y, speaker_z = get_source_position(room_x, room_y)
        speaker_x2, speaker_y2, speaker_z2 = get_source_position(room_x, room_y)
        speaker_x3, speaker_y3, speaker_z3 = get_source_position(room_x, room_y)
        while (not judge([(speaker_x, speaker_y, speaker_z), (speaker_x2, speaker_y2, speaker_z2), (speaker_x3, speaker_y3, \
            speaker_z3)], (center_x, center_y, mic_z))):
            speaker_x, speaker_y, speaker_z = get_source_position(room_x, room_y)
            speaker_x2, speaker_y2, speaker_z2 = get_source_position(room_x, room_y)
            speaker_x3, speaker_y3, speaker_z3 = get_source_position(room_x, room_y)
        # noise_x, noise_y, noise_z = get_source_position(room_x, room_y)

        room.add_source([speaker_x, speaker_y, speaker_z], signal=audio1)
        room.add_source([speaker_x2, speaker_y2, speaker_z2], signal=audio2)
        room.add_source([speaker_x3, speaker_y3, speaker_z3], signal=audio3)
        # room.add_source([noise_x, noise_y, noise_z], signal=adjusted_noise_amp)
        

        

        room.add_microphone_array(mic_locs)

        room.simulate()

        room_str = str(len_audio)
        folder_name = get_identifier() + '_' + room_str
        wav_name = folder_name + '.wav'
        txt_name = folder_name + '.txt'
        folder_name = os.path.join('/mnt/fast/nobackup/scratch4weeks/jz01019/tmp1/test_dataset', 'threespeakers', folder_name)
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
            f.write('speaker no: 3\n')
            f.write('speaker 1 position: ' + str(speaker_x) + ' ' + str(speaker_y) + ' ' + str(speaker_z) + '\n')
            f.write('speaker 2 position: ' + str(speaker_x2) + ' ' + str(speaker_y2) + ' ' + str(speaker_z2) + '\n')
            f.write('speaker 3 position: ' + str(speaker_x3) + ' ' + str(speaker_y3) + ' ' + str(speaker_z3) + '\n')
            f.write('mic position: ' + str(center_x) + ' ' + str(center_y) + ' ' + str(mic_dis) + ' ' + str(mic_z) + '\n')
            f.write('rt60: ' + str(rt60_tgt) + '\n')
            f.write('room size: ' + str(room_x) + ' ' + str(room_y) + ' ' + str(room_z) + '\n')
            # f.write('noise position: ' + str(noise_x) + ' ' + str(noise_y) + ' ' + str(noise_z) + '\n')
            # f.write('snr: ' + str(snr) + '\n')

def mix_four_audio(count):
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
 
    noise_list = get_noise_list()
    # find /Volumes/jz01019/roomacoustic/CMU_ARCTIC  -maxdepth 3 -type f -iname "*.wav"  > audio_list.txt
    # fs, audio1 = wavfile.read(audio_list[0].strip())
    for i in range(count):
        while(True):
            audio_1_index = int(get_random_number(0, len(audio_list)))
            audio_2_index = int(get_random_number(0, len(audio_list)))
            audio_3_index = int(get_random_number(0, len(audio_list)))
            audio_4_index = int(get_random_number(0, len(audio_list)))
            if (audio_1_index != audio_2_index) and (audio_1_index != audio_3_index) and (audio_1_index != audio_4_index) \
                and (audio_2_index != audio_3_index) and (audio_2_index != audio_4_index) and (audio_3_index != audio_4_index):
                break
        audio_p1 = audio_list[audio_1_index].strip()
        audio_p2 = audio_list[audio_2_index].strip()
        audio_p3 = audio_list[audio_3_index].strip()
        audio_p4 = audio_list[audio_4_index].strip()
        # audio_p = audio_p.strip()
        # fs, audio1 = wavfile.read(audio_p)
        audio1, fs = librosa.load(audio_p1, sr=16000)
        audio2, fs = librosa.load(audio_p2, sr=16000)
        audio3, fs = librosa.load(audio_p3, sr=16000)
        audio4, fs = librosa.load(audio_p4, sr=16000)
        len_audio_1 = len(audio1)
        len_audio_2 = len(audio2)
        len_audio_3 = len(audio3)
        len_audio_4 = len(audio4)

        len_audio = min(len_audio_1, len_audio_2, len_audio_3, len_audio_4)
        audio1 = audio1[:len_audio]
        audio2 = audio2[:len_audio]
        audio3 = audio3[:len_audio]
        audio4 = audio4[:len_audio]

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
        mic_locs, center_x, center_y, mic_dis, mic_z = get_mic_position(room_x, room_y)
        speaker_x, speaker_y, speaker_z = get_source_position(room_x, room_y)
        speaker_x2, speaker_y2, speaker_z2 = get_source_position(room_x, room_y)
        speaker_x3, speaker_y3, speaker_z3 = get_source_position(room_x, room_y)
        speaker_x4, speaker_y4, speaker_z4 = get_source_position(room_x, room_y)
        while (not judge([(speaker_x, speaker_y, speaker_z), (speaker_x2, speaker_y2, speaker_z2), (speaker_x3, speaker_y3, \
        speaker_z3), (speaker_x4, speaker_y4, speaker_z4)], (center_x, center_y, mic_z))):
            speaker_x, speaker_y, speaker_z = get_source_position(room_x, room_y)
            speaker_x2, speaker_y2, speaker_z2 = get_source_position(room_x, room_y)
            speaker_x3, speaker_y3, speaker_z3 = get_source_position(room_x, room_y)
            speaker_x4, speaker_y4, speaker_z4 = get_source_position(room_x, room_y)
        # noise_x, noise_y, noise_z = get_source_position(room_x, room_y)

        room.add_source([speaker_x, speaker_y, speaker_z], signal=audio1)
        room.add_source([speaker_x2, speaker_y2, speaker_z2], signal=audio2)
        room.add_source([speaker_x3, speaker_y3, speaker_z3], signal=audio3)
        room.add_source([speaker_x4, speaker_y4, speaker_z4], signal=audio4)
        # room.add_source([noise_x, noise_y, noise_z], signal=adjusted_noise_amp)
        

        

        room.add_microphone_array(mic_locs)

        room.simulate()

        room_str = str(len_audio)
        folder_name = get_identifier() + '_' + room_str
        wav_name = folder_name + '.wav'
        txt_name = folder_name + '.txt'
        folder_name = os.path.join('/mnt/fast/nobackup/scratch4weeks/jz01019/tmp1/training_dataset', 'fourspeakers', folder_name)
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
            f.write('speaker no: 4\n')
            f.write('speaker 1 position: ' + str(speaker_x) + ' ' + str(speaker_y) + ' ' + str(speaker_z) + '\n')
            f.write('speaker 2 position: ' + str(speaker_x2) + ' ' + str(speaker_y2) + ' ' + str(speaker_z2) + '\n')
            f.write('speaker 3 position: ' + str(speaker_x3) + ' ' + str(speaker_y3) + ' ' + str(speaker_z3) + '\n')
            f.write('speaker 4 position: ' + str(speaker_x4) + ' ' + str(speaker_y4) + ' ' + str(speaker_z4) + '\n')
            f.write('mic position: ' + str(center_x) + ' ' + str(center_y) + ' ' + str(mic_dis) + ' ' + str(mic_z) + '\n')
            f.write('rt60: ' + str(rt60_tgt) + '\n')
            f.write('room size: ' + str(room_x) + ' ' + str(room_y) + ' ' + str(room_z) + '\n')
            # f.write('noise position: ' + str(noise_x) + ' ' + str(noise_y) + ' ' + str(noise_z) + '\n')
            # f.write('snr: ' + str(snr) + '\n')

if __name__ == '__main__':
    mix_two_audio(30000, phrase='train')
    # mix_four_audio(1)