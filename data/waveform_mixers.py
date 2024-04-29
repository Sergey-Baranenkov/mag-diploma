import random
import sre_compile
import numpy as np
import torch
import torch.nn as nn
import pyloudnorm as pyln


# Проблема этого segment mixer в том что если классов мало, батч маленький, а mix_num большой
# То в mixture скорее всего будет несколько звуков одного и того же класса и лосс будет больше
# чем он на самом деле.

class SegmentMixer(nn.Module):
    def __init__(self, max_mix_num, lower_db, higher_db):
        super(SegmentMixer, self).__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

    def __call__(self, waveforms, names=None, mixtures=None):

        batch_size = waveforms.shape[0]

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(0, batch_size):

            segment = waveforms[n].clone()

            # create zero tensors as the background template
            noise = torch.zeros_like(segment)

            mix_num = random.randint(2, self.max_mix_num)
            assert mix_num >= 2

            for i in range(1, mix_num):
                next_segment = waveforms[(n + i) % batch_size]
                rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=segment, **self.loudness_param)
                noise += rescaled_next_segment

            # randomly normalize background noise
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            # create audio mixyure
            mixture = segment + noise

            # declipping if need be
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                segment *= 0.9 / max_value
                mixture *= 0.9 / max_value

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['mixture'], data_dict['segment']


class BalancedSegmentMixer(nn.Module):
    def __init__(self, max_mix_num, lower_db, higher_db):
        super(BalancedSegmentMixer, self).__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

    def __call__(self, waveforms, names, mixtures=None):
        unique_names_length = len(set(names))
        is_one_class_batch = unique_names_length < 2
        # print('unique_names_length:', unique_names_length, names)
        # if is_one_class_batch:
        #     print('one class batch!')

        batch_size = waveforms.shape[0]

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(0, batch_size):
            segment = waveforms[n].clone()
            segment_name = names[n]

            # create zero tensors as the background template
            noise = torch.zeros_like(segment)

            # У нас может произойти ситуация когда батч состоит только из элементов одного класса,
            # это значит что наш сигнал не содержит шума и модель должна вернуть этот сигнал как есть
            # Так как это редкая ситуация и такой юзкейс возможен, то в целом это не проблема
            if is_one_class_batch:
                mix_num = 1
            else:
                mix_num = random.randint(2, min(self.max_mix_num, unique_names_length))

            used_names = {segment_name}

            for i in range(1, mix_num):
                # Выбор следующего сегмента так, чтобы класс был уникальным
                unique_segments_indices = [idx for idx, name in enumerate(names) if name not in used_names]
                if not unique_segments_indices:
                    break  # Выход, если уникальные классы закончились

                next_idx = random.choice(unique_segments_indices)
                next_segment_name = names[next_idx]
                next_segment = waveforms[next_idx]
                used_names.add(next_segment_name)  # Добавляем имя следующего сегмента в использованные

                rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=segment, **self.loudness_param)
                noise += rescaled_next_segment

            # randomly normalize background noise
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            # create audio mixture
            mixture = segment + noise

            # declipping if need be
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                segment *= 0.9 / max_value
                mixture *= 0.9 / max_value

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['mixture'], data_dict['segment']


class PassThroughSegmentMixer(nn.Module):
    def __init__(self, **kwargs):
        super(PassThroughSegmentMixer, self).__init__()

    def __call__(self, waveforms, names, mixtures):
        batch_size = waveforms.shape[0]

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(0, batch_size):
            segment = waveforms[n].clone()
            mixture = mixtures[n].clone()

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['mixture'], data_dict['segment']


def get_segment_mixer(mixer_type: str):
    if mixer_type == 'default':
        return SegmentMixer
    if mixer_type == 'balanced':
        return BalancedSegmentMixer
    if mixer_type == 'passthrough':
        return PassThroughSegmentMixer

    raise Exception(f'no segment mixer found for type {mixer_type}')


def rescale_to_match_energy(segment1, segment2):
    ratio = get_energy_ratio(segment1, segment2)
    rescaled_segment1 = segment1 / ratio
    return rescaled_segment1


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):
    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio


def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10):
    rescaled_audio = rescale_to_match_energy(audio, reference)

    delta_loudness = random.randint(lower_db, higher_db)

    gain = np.power(10.0, delta_loudness / 20.0)

    return gain * rescaled_audio


def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise ValueError("Input must be a PyTorch tensor.")


def numpy_to_torch(array):
    """Convert a NumPy array to a PyTorch tensor."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise ValueError("Input must be a NumPy array.")


# decayed
def random_loudness_norm(audio, lower_db=-35, higher_db=-15, sr=32000):
    device = audio.device
    audio = torch_to_numpy(audio.squeeze(0))
    # randomly select a norm volume
    norm_vol = random.randint(lower_db, higher_db)

    # measure the loudness first 
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio
    normalized_audio = pyln.normalize.loudness(audio, loudness, norm_vol)

    normalized_audio = numpy_to_torch(normalized_audio).unsqueeze(0)

    return normalized_audio.to(device)
