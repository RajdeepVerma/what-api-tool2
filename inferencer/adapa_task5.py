import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models
import librosa
import pandas as pd
import requests


class Task5Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

        self.final = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2.features(x)
        x = x.max(dim=-1)[0].max(dim=-1)[0]
        x = self.final(x)
        return x


class AudioDataset(Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return self.sequence.shape[0]

    def __getitem__(self, idx):
        sample = self.sequence[idx, ...]
        i = np.random.randint(sample.shape[1])
        sample = torch.cat([
            sample[:, i:, :],
            sample[:, :i, :]],
            dim=1)
        return sample


class DcaseAdapatask5():

    def __init__(self):
        self.device = None
        self.model = None
        self.data_path = None

        try:
            self.data_path = 'data/'
            model_name = 'model_system1'
            if not os.path.isfile(self.data_path + model_name):
                print('Downloading model')
                model_file = requests.get('https://github.com/sainathadapa/'
                                 'dcase2019-task5-urban-sound-tagging/releases/download/1.0/model_system1')
                with open(self.data_path + model_name, 'wb') as file:
                    file.write(model_file.content)

            self.device = torch.device('cpu')
            self.model = Task5Model(31).to(self.device)
            self.model.load_state_dict(
                torch.load(self.data_path + model_name, map_location='cpu'))
            self.channel_means = np.load(self.data_path + 'channel_means.npy')
            self.channel_stds = np.load(self.data_path + 'channel_stds.npy')

        except Exception as error:
            print('Error Iniciando el inferenciador ' + str(error))
            raise

    def run_inferencer(self, filename, audio, samplerate):

        try:

            logmel = self.compute_melspec(audio, samplerate)
            logmel_expand = np.expand_dims(logmel.T[:635, :], axis=0)
            logmel_expand = logmel_expand[:, None, :, :]
            logmel_expand = (logmel_expand - self.channel_means) / self.channel_stds
            dataset = AudioDataset(torch.Tensor(logmel_expand))
            loader = DataLoader(dataset, 64, shuffle=False)

            all_preds = []
            for _ in range(10):
                preds = []
                for inputs in loader:
                    inputs = inputs.to(self.device)
                    with torch.set_grad_enabled(False):
                        self.model = self.model.eval()
                        outputs = self.model(inputs)
                        preds.append(outputs.detach().cpu().numpy())
                preds = np.concatenate(preds, axis=0)
                preds = (1 / (1 + np.exp(-preds)))
                all_preds.append(preds)
            tmp = all_preds[0]
            for pred in all_preds[1:]:
                tmp += pred
            tmp = tmp / 10
            preds = tmp

            output_df = pd.DataFrame(
                preds, columns=[
                    '1_engine', '2_machinery-impact', '3_non-machinery-impact',
                    '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog',
                    '1-1_small-sounding-engine', '1-2_medium-sounding-engine',
                    '1-3_large-sounding-engine', '2-1_rock-drill', '2-2_jackhammer',
                    '2-3_hoe-ram', '2-4_pile-driver', '3-1_non-machinery-impact',
                    '4-1_chainsaw', '4-2_small-medium-rotating-saw',
                    '4-3_large-rotating-saw', '5-1_car-horn', '5-2_car-alarm', '5-3_siren',
                    '5-4_reverse-beeper', '6-1_stationary-music', '6-2_mobile-music',
                    '6-3_ice-cream-truck', '7-1_person-or-small-group-talking',
                    '7-2_person-or-small-group-shouting', '7-3_large-crowd',
                    '7-4_amplified-speech', '8-1_dog-barking-whining'])
            output_df['audio_filename'] = pd.Series(filename, index=output_df.index)

            for entity in [
                '1-X_engine-of-uncertain-size', '2-X_other-unknown-impact-machinery',
                '4-X_other-unknown-powered-saw', '5-X_other-unknown-alert-signal',
                '6-X_music-from-uncertain-source', '7-X_other-unknown-human-voice']:
                output_df[entity] = 0

            cols_in_order = [
                "audio_filename", "1-1_small-sounding-engine",
                "1-2_medium-sounding-engine", "1-3_large-sounding-engine",
                "1-X_engine-of-uncertain-size", "2-1_rock-drill",
                "2-2_jackhammer", "2-3_hoe-ram", "2-4_pile-driver",
                "2-X_other-unknown-impact-machinery", "3-1_non-machinery-impact",
                "4-1_chainsaw", "4-2_small-medium-rotating-saw",
                "4-3_large-rotating-saw", "4-X_other-unknown-powered-saw",
                "5-1_car-horn", "5-2_car-alarm", "5-3_siren", "5-4_reverse-beeper",
                "5-X_other-unknown-alert-signal", "6-1_stationary-music",
                "6-2_mobile-music", "6-3_ice-cream-truck",
                "6-X_music-from-uncertain-source", "7-1_person-or-small-group-talking",
                "7-2_person-or-small-group-shouting", "7-3_large-crowd",
                "7-4_amplified-speech", "7-X_other-unknown-human-voice",
                "8-1_dog-barking-whining", "1_engine", "2_machinery-impact",
                "3_non-machinery-impact", "4_powered-saw", "5_alert-signal",
                "6_music", "7_human-voice", "8_dog"]
            output_df = output_df.loc[:, cols_in_order]

            return output_df.iloc[0]

        except Exception as error:
            print('An error was found ' + str(error))
            raise

    @staticmethod
    def compute_melspec(audio, samplerate):
        wav = librosa.resample(audio, orig_sr=samplerate, target_sr=44100)
        melspec = librosa.feature.melspectrogram(
            wav,
            sr=44100,
            n_fft=128 * 20,
            hop_length=347 * 2,
            n_mels=128,
            fmin=20,
            fmax=44100 // 2)
        logmel = librosa.core.power_to_db(melspec)
        return logmel
