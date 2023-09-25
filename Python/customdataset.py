import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from torchvision import transforms
import torchaudio.transforms as ta_transforms
import math
import torchaudio
import cv2
import cmapy
import random
import nlpaug.augmenter.audio as naa


class Biquad:

  # pretend enumeration
  LOWPASS, HIGHPASS, BANDPASS, PEAK, NOTCH, LOWSHELF, HIGHSHELF = range(7)

  def __init__(self, typ, freq, srate, Q, dbGain=0):
    types = {
      Biquad.LOWPASS : Biquad.lowpass,
      Biquad.HIGHPASS : Biquad.highpass,
      Biquad.BANDPASS : Biquad.bandpass,
      Biquad.PEAK : Biquad.peak,
      Biquad.NOTCH : Biquad.notch,
      Biquad.LOWSHELF : Biquad.lowshelf,
      Biquad.HIGHSHELF : Biquad.highshelf
    }
    assert typ in types
    self.typ = typ
    self.freq = float(freq)
    self.srate = float(srate)
    self.Q = float(Q)
    self.dbGain = float(dbGain)
    self.a0 = self.a1 = self.a2 = 0
    self.b0 = self.b1 = self.b2 = 0
    self.x1 = self.x2 = 0
    self.y1 = self.y2 = 0
    # only used for peaking and shelving filter types
    A = math.pow(10, dbGain / 40)
    omega = 2 * math.pi * self.freq / self.srate
    sn = math.sin(omega)
    cs = math.cos(omega)
    alpha = sn / (2*Q)
    beta = math.sqrt(A + A)
    types[typ](self,A, omega, sn, cs, alpha, beta)
    # prescale constants
    self.b0 /= self.a0
    self.b1 /= self.a0
    self.b2 /= self.a0
    self.a1 /= self.a0
    self.a2 /= self.a0

  def lowpass(self,A, omega, sn, cs, alpha, beta):
    self.b0 = (1 - cs) /2
    self.b1 = 1 - cs
    self.b2 = (1 - cs) /2
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def highpass(self, A, omega, sn, cs, alpha, beta):
    self.b0 = (1 + cs) /2
    self.b1 = -(1 + cs)
    self.b2 = (1 + cs) /2
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def bandpass(self, A, omega, sn, cs, alpha, beta):
    self.b0 = alpha
    self.b1 = 0
    self.b2 = -alpha
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def notch(self, A, omega, sn, cs, alpha, beta):
    self.b0 = 1
    self.b1 = -2 * cs
    self.b2 = 1
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def peak(self, A, omega, sn, cs, alpha, beta):
    self.b0 = 1 + (alpha * A)
    self.b1 = -2 * cs
    self.b2 = 1 - (alpha * A)
    self.a0 = 1 + (alpha /A)
    self.a1 = -2 * cs
    self.a2 = 1 - (alpha /A)

  def lowshelf(self, A, omega, sn, cs, alpha, beta):
    self.b0 = A * ((A + 1) - (A - 1) * cs + beta * sn)
    self.b1 = 2 * A * ((A - 1) - (A + 1) * cs)
    self.b2 = A * ((A + 1) - (A - 1) * cs - beta * sn)
    self.a0 = (A + 1) + (A - 1) * cs + beta * sn
    self.a1 = -2 * ((A - 1) + (A + 1) * cs)
    self.a2 = (A + 1) + (A - 1) * cs - beta * sn

  def highshelf(self, A, omega, sn, cs, alpha, beta):
    self.b0 = A * ((A + 1) + (A - 1) * cs + beta * sn)
    self.b1 = -2 * A * ((A - 1) + (A + 1) * cs)
    self.b2 = A * ((A + 1) + (A - 1) * cs - beta * sn)
    self.a0 = (A + 1) - (A - 1) * cs + beta * sn
    self.a1 = 2 * ((A - 1) - (A + 1) * cs)
    self.a2 = (A + 1) - (A - 1) * cs - beta * sn

  # perform filtering function
  def __call__(self, x):
    y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
    self.x2 = self.x1
    self.x1 = x
    self.y2 = self.y1
    self.y1 = y
    return y

  # provide a static result for a given frequency f
  def result(self, f):
    phi = (math.sin(math.pi * f * 2/(2*self.srate)))**2
    r =((self.b0+self.b1+self.b2)**2 - \
    4*(self.b0*self.b1 + 4*self.b0*self.b2 + \
    self.b1*self.b2)*phi + 16*self.b0*self.b2*phi*phi) / \
    ((1+self.a1+self.a2)**2 - 4*(self.a1 + 4*self.a2 + \
    self.a1*self.a2)*phi + 16*self.a2*phi*phi)
    if(r < 0):
      r = 0
    return r**(.5)

  # provide a static log result for a given frequency f
  def log_result(self, f):
    try:
      r = 20 * math.log10(self.result(f))
    except:
      r = -200
    return r

  # return computed constants
  def constants(self):
    return self.a1, self.a2, self.b0, self.b1, self.b2

  def __str__(self):
    return "Type:%d,Freq:%.1f,Rate:%.1f,Q:%.1f,Gain:%.1f" % (self.typ,self.freq,self.srate,self.Q,self.dbGain)


class CustomDataset(Dataset):
    def __init__(self, path, txt_list,
                 sample_rate=4000,
                 hop_length=40,
                 n_mels=128,
                 n_fft=1024,
                 win_length=800,
                 augmentation=False,
                 filter_params=False,
                 padding_type=0,
                 freq_mask=False,
                 time_mask=False,
                 multi_channels=False,
                 clipping=False,
                 target_size=(300, 300),
                 th=5):
        self.path = path
        self.txt_list = txt_list

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length

        self.augmentation = augmentation
        self.filter_params = filter_params
        self.padding_type = padding_type
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.multi_channels = multi_channels
        self.clipping = clipping
        self.target_size = target_size
        self.th = int(th * self.sample_rate / self.hop_length)

        self.get_file_list()

        self.delete_list = []
        self.x, self.y = self.get_mel_spectrogram()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_file_list(self):
        # self.heas = []
        self.wavs = []
        self.tsvs = []

        for path_txt in self.txt_list:
            with open(path_txt, "r") as f:
                P_id, n, sr = f.readline().split()
                for _ in range(int(n)):
                    _, hea, wav, tsv = f.readline().split()
                    # self.heas.append(hea)
                    self.wavs.append(wav)
                    self.tsvs.append(tsv)
        # self.heas.sort()
        self.wavs.sort()
        self.tsvs.sort()

    def apply_filter(self, audio):
        for filter_param in self.filter_params:
            audio = self.filter_torchaudio(audio, filter_param)
        return audio

    # torchaudio로 필터링 적용
    def filter_torchaudio(self, _audio, _params):
        biquad_filter = Biquad(*_params)
        a1, a2, b0, b1, b2 = biquad_filter.constants()
        _filtered_audio = torchaudio.functional.biquad(
            waveform=_audio,
            b0=b0,
            b1=b1,
            b2=b2,
            a0=1.0,
            a1=a1,
            a2=a2
        )
        return _filtered_audio

    def zero_padding(self, spec, target_length, pad_position=0):
        pad_width = target_length - spec.shape[-1]
        # 뒷부분에 zero padding
        if pad_position < 0.5:
            padded_spec = torch.nn.functional.pad(spec, (0, pad_width, 0, 0), "constant", 0)
        # 앞부분에 zero padding
        else:
            padded_spec = torch.nn.functional.pad(spec, (pad_width, 0, 0, 0), "constant", 0)
        return padded_spec

    def another_padding(self, spec, target_length, types, df, _iter):
        pad_width = target_length - spec.shape[-1]
        prob = random.random()
        padded_spec = self.zero_padding(spec, target_length, prob)

        copied_rows = df.iloc[0:, :]
        padded_rows = copied_rows.copy()
        original_size = self.th / (self.sample_rate / self.hop_length) * _iter
        pad_width_df = pad_width / self.sample_rate

        # 원본 복사
        if types == 1:
            aug = spec
        # 원본에 어그멘테이션 적용
        else:
            aug = self.gen_augmented(spec, padding_mode=True)
        # 뒷부분에 padding
        if prob < 0.5:
            padded_spec[:, padded_spec.shape[-1] - pad_width:] = aug[:, :pad_width]
            padded_rows[0] += (original_size - pad_width_df)
            padded_rows[1] += (original_size - pad_width_df)
            result_df = pd.concat([df, padded_rows], axis=0)
        # 앞부분에 padding
        else:
            padded_spec[:, :pad_width] = aug[:, :pad_width]
            top_rows = copied_rows[(copied_rows[0] >= 0) & (copied_rows[0] < pad_width_df)].copy()
            top_rows[1] = top_rows[1].clip(0, pad_width_df)
            padded_rows[0] += pad_width_df
            padded_rows[1] += pad_width_df
            result_df = pd.concat([top_rows, padded_rows], axis=0)
        return padded_spec, result_df

    def gen_augmented(self, spec, padding_mode):
        augment_list = [naa.NoiseAug(),
                        naa.LoudnessAug(factor=(0.5, 2)),
                        naa.PitchAug(sampling_rate=self.sample_rate, factor=(-1, 3))
                        ]
        # 패딩에 augmentation 사용하는 경우
        if padding_mode is True:
            aug_idx = random.randint(0, len(augment_list) - 1)
            augmented_data = augment_list[aug_idx].augment(spec.numpy()[0])
            augmented_data = torch.from_numpy(np.array(augmented_data))
            return augmented_data
        # 데이터에 augmentation 사용하는 경우
        else:
            augmented_data_list = [spec]
            for aug in augment_list:
                augmented_data = aug.augment(spec.numpy()[0])
                augmented_data = torch.from_numpy(np.array(augmented_data))
                augmented_data_list.append(augmented_data)
            return augmented_data_list

    def padded_df(self, df, pad_position, pad_width, _iter):
        copied_rows = df.iloc[0:, :]
        padded_rows = copied_rows.copy()
        original_size = self.th / (self.sample_rate / self.hop_length) * _iter
        # print(original_size, _iter, pad_width)
        # 패딩 위치가 뒤
        if pad_position < 0.5:
            padded_rows[0] += (original_size - pad_width)
            padded_rows[1] += (original_size - pad_width)
            result_df = pd.concat([df, padded_rows], axis=0)
        # 패딩 위치가 앞
        else:
            top_rows = copied_rows[(copied_rows[0] >= 0) & (copied_rows[0] < pad_width)].copy()
            top_rows[1] = top_rows[1].clip(0, pad_width)
            padded_rows[0] += pad_width
            padded_rows[1] += pad_width
            result_df = pd.concat([top_rows, padded_rows], axis=0)
        return result_df

    def process_label(self, tsv_data, _iter):
        label = []
        for _, tsv_row in tsv_data.iterrows():
            # S1, S2에 속한다면
            if tsv_row[2] in [1, 3]:
                # 구간 불러와서 sr곱하고 hop_length로 나누기
                tsv_row[0] = tsv_row[0] * self.sample_rate / self.hop_length - (_iter * self.th)
                tsv_row[1] = tsv_row[1] * self.sample_rate / self.hop_length - (_iter * self.th)
                tsv_row[2] = 1 if tsv_row[2] == 1 else 2 # S1=1, S2=2
                # 시작점 혹은 끝점이 구간 안에 존재한다면
                if (0 <= tsv_row[0] < self.th or \
                    0 < tsv_row[1] <= self.th):
                    # 시작점이 0보다 작은 경우 0으로
                    if tsv_row[0] < 0:
                        tsv_row[0] = 0
                    # 끝점이 구간보다 큰 경우 구간의 끝점으로
                    if tsv_row[1] > self.th:
                        tsv_row[1] = self.th
                    # 최종 resize한 값으로 스케일링
                    tsv_row[0] *= (self.target_size[1] - 1) / self.th
                    tsv_row[1] *= (self.target_size[1] - 1) / self.th
                    label.append([tsv_row[0] / self.target_size[1], 0,
                                tsv_row[1] / self.target_size[1], 1,
                                int(tsv_row[2])])
                # 시작점 혹은 끝점이 구간 안에 존재하지 않는다면
                else: continue
            # S1, S2에 속하면서 시작점 혹은 끝점이 구간 안에 존재하는 경우를 제외한 나머지 경우
            else: continue
        return label

    def masking(self, spec, label):
        # Freqeuncy masking
        if self.freq_mask != False:
            # 정해진 확률에 속할 경우 masking
            if random.random() <= self.freq_mask[0]:
                    spec = ta_transforms.FrequencyMasking(freq_mask_param=self.freq_mask[1] * spec.shape[1])(spec)
            masked_label = label.copy()
        # Time masking
        if self.time_mask != False:
            # 정해진 확률에 속할 경우 masking
            if random.random() <= self.time_mask[0]:
                masked_label = []
                orig_spec = spec
                spec = ta_transforms.TimeMasking(time_mask_param=self.time_mask[1] * spec.shape[-1])(spec)
                # 마스킹 부분 라벨 제거
                if self.time_mask[2] is True:
                    # 일반적인 경우
                    try:
                        start = np.min(np.where(spec != orig_spec)[-1]) / spec.shape[-1]
                        end = np.max(np.where(spec != orig_spec)[-1]) / spec.shape[-1]
                        # print(start, end)
                        for l in label:
                            # 라벨이 마스크 영역보다 큰 경우
                            if start > l[0] and l[2] > end:
                                masked_label.append([l[0], l[1], start, l[3], l[4]])
                                masked_label.append([end, l[1], l[2], l[3], l[4]])
                            # 라벨의 끝점이 마스크 영역 안에 있는 경우
                            elif l[0] < start and start <= l[2]:
                                l[2] = start
                                masked_label.append(l)
                            # 라벨의 시작점이 마스크 영역 안에 있는 경우
                            elif l[0] <= end and end < l[2]:
                                l[0] = end
                                masked_label.append(l)
                            # 그 외의 모든 경우 중 라벨이 마스크 보다 작지 않은 경우
                            elif not(start < l[0] and l[2] < end):
                                masked_label.append(l)
                    # 제로패딩 영역에 마스킹이 들어가 라벨에 변경이 없을 경우
                    except:
                        masked_label = label.copy()
                # 라벨 보존
                else:
                    masked_label = label.copy()
            # 확률에 속하지 않아 masking을 실시하지 않은 경우
            else:
                masked_label = label.copy()
        return spec, masked_label

    def normalize_spectrogram(self, spec):
        normalized = (spec-spec.min()) / (spec.max() - spec.min())
        return normalized

    def change_channels(self, spec):
        spec *= 255
        spec = np.array(spec[0])
        spec = spec.astype(np.uint8)
        spec = cv2.applyColorMap(spec.astype(np.uint8), cmapy.cmap('magma'))
        spec = transforms.ToTensor()(spec)
        return spec

    def blank_clipping(self, img):
        img[img < 10/255] = 0
        img = np.transpose(np.array(img), (1, 2, 0))  # 텐서 > 넘파이
        # 3채널 이미지의 경우
        if self.multi_channels is True:
            copy = img.copy()   # 사본 생성
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)   # 흑백으로
        # 1채널 이미지의 경우
        else:
            copy = img
        # 행별로 black_percent 계산
        for row in range(img.shape[0] - 1, 0, -1):
            black_percent = len(np.where(img[row,:]==0)[0])/len(img[row,:])
            print(black_percent)
            if black_percent < 0.80:
                break
        # # clipping
        # if (row - 1) > 0:
        #     copy = copy[:(row - 1), :, :]
        # print(row)
        row = (row + 1) * self.target_size[0] / img.shape[0]
        # print(row)
        return transforms.ToTensor()(copy), row

    def resize_spectrogram(self, spec, new_shape):
        resized_spec = transforms.functional.resize(img=spec, size=new_shape, antialias=None)
        return resized_spec

    def process_data(self, spec, label):
        # 멜스펙트로그램 변환
        spec = ta_transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                        n_fft=self.n_fft,
                                        win_length=self.win_length,
                                        n_mels=self.n_mels,
                                        hop_length=self.hop_length)(spec)

        spec = torchaudio.functional.amplitude_to_DB(spec, multiplier=10.,
                                        amin=1e-10,
                                        db_multiplier=1.0,
                                        top_db=80.0)

        # 0~1로 정규화
        spec = self.normalize_spectrogram(spec)
        # 3채널 기능
        if self.multi_channels is True:
            spec = self.change_channels(spec)
        # Blank region clipping
        if self.clipping is True:
            spec, blank_row = self.blank_clipping(spec)
        else: blank_row = self.target_size[0]

        # Masking
        if self.freq_mask != False or self.time_mask != False:
            spec, label = self.masking(spec, label)

        # 최종 Resizing
        spec = self.resize_spectrogram(spec, self.target_size)
        return spec, label, blank_row

    def get_mel_spectrogram(self):
        audio_list = []
        labels = []
        for wav, tsv in zip(self.wavs, self.tsvs):
            # Torchaudio 이용하여 wav파일 로드
            path_wav = os.path.join(self.path, wav)
            d, org_sr = torchaudio.load(path_wav)
            d = torchaudio.functional.resample(d, orig_freq=org_sr, new_freq=self.sample_rate)
            # tsv파일 로드
            path_tsv = os.path.join(self.path, tsv)
            tsv_data = pd.read_csv(path_tsv, sep='\t', header=None)

            # 데이터 어그멘테이션
            if self.augmentation is True:
                data_list = self.gen_augmented(d, padding_mode=False)
            else:
                data_list = [d]
            for x in data_list:
                # Filtering
                if self.filter_params != False:
                    x = self.apply_filter(x)
                # 구간 = i * frame_offset: i * frame_offset + num_frames
                frame_offset, num_frames = self.th * self.hop_length, self.th * self.hop_length
                num_splits = math.ceil(x.shape[-1] / num_frames)  # 나눌 개수
                # 오디오 파일이 num_splits * num_frames보다 짧을 경우
                if x.shape[-1] < num_splits * num_frames:
                    # Zero padding
                    if self.padding_type == 0:
                        x = self.zero_padding(x, num_splits * num_frames, self.padding_type)
                    # Another padding(1: copy & paste, 2: augmentation)
                    else:
                        x, tsv_data = self.another_padding(x, num_splits * num_frames,
                                                           self.padding_type,
                                                           tsv_data, num_splits)
                # 오디오 파일이 num_frames보다 긴 경우 split
                if x.shape[-1] > num_frames:
                    for i in range(num_splits):
                        label = self.process_label(tsv_data, i)
                        # label에 아무것도 들어있지 않다면
                        if len(label) == 0:
                            continue
                        # 오디오 split
                        split = x[:, i * frame_offset:i * frame_offset + num_frames]
                        # 멜스펙트로그램, 정규화, 채널 수 조정, 클리핑, 리사이징
                        split, label, blank_row = self.process_data(split, label)
                        # 라벨에 blank_region clipping 적용
                        label = [[x1, y1, x2, blank_row / self.target_size[0], cls] for x1, y1, x2, _, cls in label]
                        labels.append(label)
                        audio_list.append(split)
                    # break
                # 원본 wav의 길이가 num_frames와 같다면(패딩을 했기에 짧을 수는 없음) split X
                else:
                    label = self.process_label(tsv_data, i)
                    # label에 아무것도 들어있지 않다면
                    if len(label) == 0:
                        continue
                    x, label, blank_row = self.process_data(x, label)
                    # 라벨에 blank_region clipping 적용
                    label = [[x1, y1, x2, blank_row, cls] for x1, y1, x2, _, cls in label]
                    labels.append(label)
                    audio_list.append(x)
                    # break
        # print(len(audio_list))
        # print(len(labels))
        return torch.stack(audio_list), labels