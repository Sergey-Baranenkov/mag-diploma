import os

from pydub import AudioSegment
import json
import random

from pipeline import separate_audio
import shutil
from ipywidgets import widgets


def mix_wav_files(files, output_path):
    mixed_sound = AudioSegment.from_file(files[0])

    for file in files[1:]:
        next_sound = AudioSegment.from_file(file)
        mixed_sound = mixed_sound.overlay(next_sound)

    mixed_sound.export(output_path, format='wav')


def select_random_elements(data, n_elements, seed=None):
    if seed is not None:
        random.seed(seed)

    n_elements = min(n_elements, len(data))

    selected_elements = random.sample(data, n_elements)
    return selected_elements


def separate_and_visualize(mix_cnt, seeds, model, device, model_name):
    seed_mixtures = {}

    with open('../datafiles/dota2.val.json') as file:
        parsed_json = json.load(file)['data']
        for seed in seeds:
            random_mixes = select_random_elements(parsed_json, mix_cnt, seed)
            seed_mixtures[seed] = {
                'test_files': [f'../{mix["wav"]}' for mix in random_mixes],
                'classes': [mix['caption'] for mix in random_mixes]
            }

    mixture_dir = 'dota2_mixtures'
    if not os.path.exists(mixture_dir):
        os.makedirs(mixture_dir)

    for seed in seeds:
        mixture_path = f'{mixture_dir}/{mix_cnt}_{seed}.wav'
        mix_wav_files(seed_mixtures[seed]['test_files'], mixture_path)
        seed_mixtures[seed]['mixture_path'] = mixture_path

    audio_widgets_list = []

    for seed in seeds:
        file = seed_mixtures[seed]['mixture_path']
        classes = seed_mixtures[seed]['classes']
        audio_widgets = {cls: {} for cls in classes + ['original']}

        filename = file.split(os.sep)[-1]
        audio_widget = widgets.Audio(value=open(file, "rb").read(), format="wav", controls=True, autoplay=False)
        audio_widgets['original'][filename] = audio_widget
        output_dir = f'../separation_result/dota2/{model_name}/{filename.split(".wav")[0]}/'
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        for cls in classes:
            output_file = os.path.join(output_dir, f'{cls}.wav')
            separate_audio(model, file, cls, output_file, device, use_chunk=False)
            audio_widget = widgets.Audio(value=open(output_file, "rb").read(), format="wav", controls=True,
                                         autoplay=False)
            audio_widgets[cls][filename] = audio_widget

        audio_widgets_list.append(audio_widgets)
        shutil.copyfile(file, f'{output_dir}/original.wav')

    from utils import plot_separation_result

    for audio_widgets in audio_widgets_list:
        plot_separation_result(audio_widgets)
