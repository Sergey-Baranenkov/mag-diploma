dota2_heroes = [
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

musdb_ckpt = 'checkpoints/final/musdb18/lora_embeddings/final.ckpt'
desed_ckpt = 'checkpoints/final/desed/lora_embeddings/final.ckpt'
dota2_ckpt = 'checkpoints/final/dota2/lora_embeddings/final.ckpt'

class_checkpoint_combinations = [
    ('vacuum cleaner', desed_ckpt),
    ('frying', desed_ckpt),
    ('dishes', desed_ckpt),
    ('blender', desed_ckpt),
    ('speech', desed_ckpt),
    ('dog', desed_ckpt),
    ('alarm bell ringing', desed_ckpt),
    ('running water', desed_ckpt),
    ('electric shaver toothbrush', desed_ckpt),
    ('cat', desed_ckpt),

    ('drums', musdb_ckpt),
    ('bass', musdb_ckpt),
    ('vocal', musdb_ckpt),
    ('other musical instruments', musdb_ckpt),
]

class_checkpoint_combinations += [(hero, dota2_ckpt) for hero in dota2_heroes]
