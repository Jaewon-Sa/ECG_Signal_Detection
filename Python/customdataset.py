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
        self.multi_channels = multi_channels
        self.clipping = clipping
        self.target_size = target_size
        self.th = int(th * self.sample_rate / self.hop_length)

        self.get_file_list()

        self.delete_list = []
        self.x = self.get_mel_spectrogram()
        self.y = self.get_label()
        self.delete_data()

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
            if black_percent < 0.80:
                break
        # # clipping
        # if (row - 1) > 0:
        #     copy = copy[:(row - 1), :, :]
        # print(row)
        row = row * self.target_size[0] / img.shape[0]
        self.blank_row_list.append(row)
        # print(row)
        return transforms.ToTensor()(copy)

    def padding(self, spec, target_length, types):
        if types == 0:
            padded_spec = self.zero_padding(spec, target_length, types)
        else:
            padded_spec = self.another_padding(spec, target_length, types)
        return padded_spec

    def zero_padding(self, spec, target_length, pad_position=0):
        pad_width = target_length - spec.shape[-1]
        # 뒷부분에 zero padding
        if pad_position < 0.5:
            padded_spec = torch.nn.functional.pad(spec, (0, pad_width, 0, 0), "constant", 0)
        # 앞부분에 zero padding
        else:
            padded_spec = torch.nn.functional.pad(spec, (pad_width, 0, 0, 0), "constant", 0)
        return padded_spec

    def another_padding(self, spec, target_length, types):
        pad_width = target_length - spec.shape[-1]
        prob = random.random()
        padded_spec = self.zero_padding(spec, target_length, prob)
        if types == 1:
            aug = spec
        else:
            aug = self.gen_augmented(spec, padding_mode=True)
        # 뒷부분에 padding
        if prob < 0.5:
            padded_spec[:, padded_spec.shape[-1] - pad_width:] = aug[:, :pad_width]
        # 앞부분에 padding
        else:
            padded_spec[:, :pad_width] = aug[:, :pad_width]
        pad_width /= self.sample_rate
        # 패딩 위치(뒤: prob<0.5, 앞: prob>=0.5), 패딩 길이(단위: 초)
        self.padding_list.append((prob, pad_width))
        return padded_spec

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

    def resize_spectrogram(self, spec, new_shape):
        resized_spec = transforms.functional.resize(img=spec, size=new_shape, antialias=None)
        return resized_spec

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

    def processing(self, spec):
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
            spec = self.blank_clipping(spec)
        # 최종 Resizing
        spec = self.resize_spectrogram(spec, self.target_size)
        return spec

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

    def get_mel_spectrogram(self):
        audio_list = []
        self.iter_list = []
        self.blank_row_list = []
        self.padding_list = []

        for path_wav in self.wavs:
            path = os.path.join(self.path, path_wav)
            # Torchaudio 이용하여 파일 로드
            d, org_sr = torchaudio.load(path)
            d = torchaudio.functional.resample(d, orig_freq=org_sr, new_freq=self.sample_rate)

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
                    # Padding
                    x = self.padding(x, num_splits * num_frames, self.padding_type)
                else: self.padding_list.append(0)
                # 오디오 파일이 num_frames보다 긴 경우
                if x.shape[-1] > num_frames:
                    self.iter_list.append(num_splits)
                    for i in range(num_splits):
                        split = x[:, i * frame_offset:i * frame_offset + num_frames]
                        # 멜스펙트로그램, 정규화, 채널 수 조정, 클리핑, 리사이징
                        split = self.processing(split)
                        audio_list.append(split)
                    # break
                # 원본 wav의 길이가 num_frames보다 짧거나 같다면
                else:
                    self.iter_list.append(1)
                    # 멜스펙트로그램, 정규화, 채널 수 조정, 클리핑, 리사이징
                    x = self.processing(x)
                    audio_list.append(x)
                    # break
        # print(len(audio_list), len(self.iter_list), len(self.blank_row_list), len(self.padding_list))
        return torch.stack(audio_list)

    def get_label(self):
        labels = []
        idx = 0
        for i, path_tsv in enumerate(self.tsvs):
            path = os.path.join(self.path, path_tsv)
            orig_tsv_data = pd.read_csv(path, sep='\t', header=None)
            # Augmentation을 사용한다면 4회 반복, 아니라면 1회
            for j in range(4 if self.augmentation is True else 1):
                # Augmentation 유무에 따라 사용할 _idx 변경
                if self.augmentation is True:
                    _idx = i * 4 + j
                else:
                    _idx = i
                    tsv_data = orig_tsv_data.copy()
                iter = self.iter_list[_idx]
                # self.padding_type이 0이 아닌 경우(1, 2인 경우)
                if self.padding_type != 0 and self.padding_list[_idx] != 0:
                    # 패딩 위치(앞: 0, 뒤: 1), 패딩 길이(단위: 초)
                    pad_info = self.padding_list[_idx]
                    tsv_data = self.padded_df(orig_tsv_data, pad_info[0], pad_info[1], iter)
                # 한 이미지당 split의 개수만큼 반복
                for _iter in range(iter):
                    label = []
                    # blank clipping row 불러오기
                    if self.clipping is True:
                        blank_row = self.blank_row_list[sum(self.iter_list[:_idx]) + _iter]
                    else:
                        blank_row = self.target_size[0]
                    for _, tsv_row in tsv_data.iterrows():
                        # S1, S2에 속한다면
                        if tsv_row[2] in [1, 3]:
                            # 구간 불러와서 sr값 곱하고 hop_legth로 나누기
                            tsv_row[0] = tsv_row[0] * self.sample_rate / self.hop_length - (_iter * self.th)
                            tsv_row[1] = tsv_row[1] * self.sample_rate / self.hop_length - (_iter * self.th)
                            tsv_row[2] = 1 if tsv_row[2] == 1 else 2    # S1=1, S2=2
                            # 시작점 혹은 끝점이 구간 안에 존재한다면
                            if (0 <= tsv_row[0] < self.th or \
                                0 < tsv_row[1] <= self.th):
                                # 시작점이 0보다 작은 경우 0으로
                                if tsv_row[0] < 0:
                                    tsv_row[0] = 0
                                # 끝점이 구간보다 큰 경우 구간의 끝점으로
                                if tsv_row[1] > self.th:
                                    tsv_row[1] = self.th
                                # 최종 resize한 값 으로 보간
                                tsv_row[0] *= (self.target_size[1] - 1) / self.th
                                tsv_row[1] *= (self.target_size[1] - 1) / self.th
                                label.append([tsv_row[0] / self.target_size[1], 0 / self.target_size[0],
                                    tsv_row[1] / self.target_size[1], blank_row / self.target_size[0],
                                    int(tsv_row[2])])# xmin, ymin, xmax, ymax, cls
                            # 시작점 혹은 끝점이 구간 안에 존재하지 않는다면
                            else: continue
                        # S1, S2에 속하면서 시작점 혹은 끝점이 구간 안에 존재하는 경우를 제외한 나머지 경우
                        else: continue
                    if(len(label)==0):
                        self.delete_list.append(idx)
                    idx += 1
                    labels.append(label)
            # break
        # print(len(labels))
        return labels

    def delete_data(self):
        delete_count=0
        for i in self.delete_list:
            del self.y[i-delete_count]
            delete_count+=1
        self.x = self.x[[i for i in range(self.x.size(0)) if i not in self.delete_list]]