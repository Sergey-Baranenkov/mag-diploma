import requests
from bs4 import BeautifulSoup
import os
import asyncio

from pydub import AudioSegment

from constants.index_classes import dota2_heroes


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


heroes = dota2_heroes

max_sounds_per_hero = 20

@background
def download_audio_from_page(url, caption):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    audio_tags = soup.findAll('audio')

    if not audio_tags:
        print(f"No audio found on {url}")
        return

    files_dir = f'dota2/{caption}'
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)

    for idx, audio in enumerate(audio_tags[:max_sounds_per_hero]):
        audio_src = audio.find('source').get('src')
        audio_response = requests.get(audio_src)
        mp3_audio_path = os.path.join(files_dir, f'{idx}.mp3')
        wav_audio_path = os.path.join(files_dir, f'{idx}.wav')

        with open(mp3_audio_path, 'wb') as f:
            f.write(audio_response.content)

        # Convert MP3 to WAV
        sound = AudioSegment.from_mp3(mp3_audio_path)
        sound.export(wav_audio_path, format="wav")
        os.remove(mp3_audio_path)


def crawl_category_pages():
    for idx, hero in enumerate(heroes):
        full_url = f'https://dota2.fandom.com/wiki/{hero}/Responses'
        download_audio_from_page(full_url, hero)


crawl_category_pages()