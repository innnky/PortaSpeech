import os
import re
import random
import json
from collections import defaultdict

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.stats import betabinom
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

import audio as Audio
from model import PreDefinedEmbedder
from utils.tools import plot_embedding, word_level_subdivision


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.preprocess_config = preprocess_config
        self.multi_speaker = model_config["multi_speaker"]
        self.in_dir = preprocess_config["path"]["raw_path"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.sort_data = preprocess_config["preprocessing"]["sort_data"]
        self.sub_divide_word = preprocess_config["preprocessing"]["text"]["sub_divide_word"]
        self.max_phoneme_num = preprocess_config["preprocessing"]["text"]["max_phoneme_num"]
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["aligner"]["beta_binomial_scaling_factor"]
        self.flist_path = preprocess_config["path"]["flist_path"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.val_prior = self.val_prior_names(
            os.path.join(self.out_dir, "val.txt"))
        self.speaker_emb = None
        if self.multi_speaker and preprocess_config["preprocessing"]["speaker_embedder"] != "none":
            self.speaker_emb = PreDefinedEmbedder(preprocess_config)
            self.speaker_emb_dict = self._init_spker_embeds(self.in_sub_dirs)

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def val_prior_names(self, val_prior_path):
        val_prior_names = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_prior_names.add(m.split("|")[0])
            return list(val_prior_names)
        else:
            return None

    def build_from_path(self):
        embedding_dir = os.path.join(self.out_dir, "spker_embed")
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs(
            (os.path.join(self.out_dir, "phones_per_word")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "attn_prior")), exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)

        print("Processing Data ...")
        out = list()
        train = list()
        val = list()
        n_frames = 0
        max_seq_len = -float('inf')
        mel_frame_len_dict = dict()

        skip_speakers = set()
        for embedding_name in os.listdir(embedding_dir):
            skip_speakers.add(embedding_name.split("-")[0])

        # Preprocess
        speakers = {}
        spk_label_dict = defaultdict(list)
        current_spk_id = 0
        for line in open(self.flist_path).readlines():
            spk, id_, phones, durations, pitch, energy = line.strip().split("|")
            spk_label_dict[spk].append(line)
            if spk not in speakers.keys():
                speakers[spk] = current_spk_id
                current_spk_id += 1


        for speaker, i  in tqdm(speakers.items()):
            save_speaker_emb = self.speaker_emb is not None and speaker not in skip_speakers
            if os.path.isdir(os.path.join(self.in_dir, speaker)):
                for line in tqdm(spk_label_dict[speaker]):
                    _, basename, phones_with_word_boundary, durations, pitch, energy = line.strip().split("|")

                    ret = self.process_utterance(
                        phones_with_word_boundary, durations, speaker, basename, save_speaker_emb)
                    if ret is None:
                        continue
                    else:
                        info, n, spker_embed = ret

                    if self.val_prior is not None:
                        if basename not in self.val_prior:
                            train.append(info)
                        else:
                            val.append(info)
                    else:
                        out.append(info)
                    if save_speaker_emb:
                        self.speaker_emb_dict[speaker].append(spker_embed)

                    if n > max_seq_len:
                        max_seq_len = n

                    n_frames += n
                    mel_frame_len_dict[basename] = n

                # Calculate and save mean speaker embedding of this speaker
                if save_speaker_emb:
                    spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
                    np.save(os.path.join(self.out_dir, 'spker_embed', spker_embed_filename),
                            np.mean(self.speaker_emb_dict[speaker], axis=0), allow_pickle=False)

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "max_seq_len": max_seq_len
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        if self.speaker_emb is not None:
            print("Plot speaker embedding...")
            plot_embedding(
                self.out_dir, *self.load_embedding(embedding_dir),
                self.divide_speaker_by_gender(self.corpus_dir), filename="spker_embed_tsne.png"
            )

        if self.val_prior is not None:
            assert len(out) == 0
            random.shuffle(train)
            train = [r for r in train if r is not None]
            val = [r for r in val if r is not None]
        else:
            assert len(train) == 0 and len(val) == 0
            random.shuffle(out)
            out = [r for r in out if r is not None]
            train = out[self.val_size:]
            val = out[: self.val_size]

        if self.sort_data:
            train.sort(key=lambda x: mel_frame_len_dict[x.split("|")[0]])
            val.sort(key=lambda x: mel_frame_len_dict[x.split("|")[0]])

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val:
                f.write(m + "\n")

        return out

    def get_label(self,  phones_with_word_boundary):
        phones_with_word_boundary = phones_with_word_boundary.split(" ")
        phones_per_word = []
        phones_count = 0
        phones = []
        for ph in phones_with_word_boundary:
            if ph not in ["^"]:
                phones.append(ph)
                phones_count += 1
            else:
                phones_per_word.append(phones_count)
                phones_count = 0
        assert len(phones) == sum(phones_per_word)
        return phones, phones_per_word

    def process_utterance(self, phones_with_word_boundary, durations, speaker, basename, save_speaker_emb):
        wav_path = os.path.join(self.in_dir, speaker,
                                "{}.wav".format(basename))

        duration = [int(d) for d in durations.split(" ")]
        phone, phones_per_word = self.get_label(phones_with_word_boundary)
        assert len(duration) == len(phone)
        if self.sub_divide_word:
            phones_per_word = word_level_subdivision(
                phones_per_word, self.max_phoneme_num)
        text = " ".join(phone)
        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav.astype(np.float32)
        spker_embed = self.speaker_emb(wav) if save_speaker_emb else None


        # Compute mel-scale spectrogram
        mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.STFT)
        assert  abs(sum(duration)-mel_spectrogram.shape[1]) < 3
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]

        # Compute alignment prior
        attn_prior = self.beta_binomial_prior_distribution(
            mel_spectrogram.shape[1],
            len(duration),
            self.beta_binomial_scaling_factor,
        )

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        attn_prior_filename = "{}-attn_prior-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "attn_prior", attn_prior_filename), attn_prior)

        phones_per_word_filename = "{}-phones_per_word-{}.npy".format(
            speaker, basename)
        np.save(os.path.join(self.out_dir, "phones_per_word",
                phones_per_word_filename), phones_per_word)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, "text"]),
            mel_spectrogram.shape[1],
            spker_embed,
        )

    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)


    def get_sub_divided_phones_per_word(self, phones_per_word, max_phoneme_num):
        res = []
        for l in phones_per_word:
            if l <= max_phoneme_num:
                res.append(l)
            else:
                s, r = l//max_phoneme_num, l%max_phoneme_num
                res += [max_phoneme_num]*s + ([r] if r else [])
        return res

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line:
                    continue
                parts = [p.strip()
                         for p in re.sub(' +', ' ', (line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def load_embedding(self, embedding_dir):
        embedding_path_list = [_ for _ in Path(embedding_dir).rglob('*.npy')]
        embedding = None
        embedding_speaker_id = list()
        # Gather data
        for path in tqdm(embedding_path_list):
            embedding = np.concatenate((embedding, np.load(path)), axis=0) \
                if embedding is not None else np.load(path)
            embedding_speaker_id.append(
                str(str(path).split('/')[-1].split('-')[0]))
        return embedding, embedding_speaker_id
