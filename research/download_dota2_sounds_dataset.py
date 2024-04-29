import requests
from bs4 import BeautifulSoup
import os
import asyncio

from pydub import AudioSegment


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


heroes = [
    "Abaddon", "Alchemist", "Ancient Apparition", "Anti-Mage", "Arc Warden", "Axe", "Bane", "Batrider",
    "Beastmaster", "Bloodseeker", "Bounty Hunter", "Brewmaster", "Bristleback", "Broodmother",
    "Centaur Warrunner", "Chaos Knight", "Chen", "Clinkz", "Clockwerk", "Crystal Maiden", "Dark Seer",
    "Dark Willow", "Dawnbreaker", "Dazzle", "Death Prophet", "Disruptor", "Doom", "Dragon Knight",
    "Drow Ranger", "Earth Spirit", "Earthshaker", "Elder Titan", "Ember Spirit", "Enchantress", "Enigma",
    "Faceless Void", "Grimstroke", "Gyrocopter", "Hoodwink", "Huskar", "Invoker", "Io", "Jakiro",
    "Juggernaut", "Keeper of the Light", "Kunkka", "Legion Commander", "Leshrac", "Lich", "Lifestealer",
    "Lina", "Lion", "Lone Druid", "Luna", "Lycan", "Magnus", "Marci", "Mars", "Medusa", "Meepo", "Mirana",
    "Monkey King", "Morphling", "Muerta", "Naga Siren", "Nature's Prophet", "Necrophos", "Night Stalker",
    "Nyx Assassin", "Ogre Magi", "Omniknight", "Oracle", "Outworld Destroyer", "Pangolier", "Phantom Assassin",
    "Phantom Lancer", "Phoenix", "Primal Beast", "Puck", "Pudge", "Pugna", "Queen of Pain", "Razor", "Riki",
    "Rubick", "Sand King", "Shadow Demon", "Shadow Fiend", "Shadow Shaman", "Silencer", "Skywrath Mage",
    "Slardar", "Slark", "Snapfire", "Sniper", "Spectre", "Spirit Breaker", "Storm Spirit", "Sven", "Techies",
    "Templar Assassin", "Terrorblade", "Tidehunter", "Timbersaw", "Tinker", "Tiny", "Treant Protector",
    "Troll Warlord", "Tusk", "Underlord", "Undying", "Ursa", "Vengeful Spirit", "Venomancer", "Viper", "Visage",
    "Void Spirit", "Warlock", "Weaver", "Windranger", "Winter Wyvern", "Witch Doctor", "Wraith King", "Zeus"
]

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