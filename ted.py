from datasets import load_dataset

LANG_MAPPER = {
    'en': 'en_us',
    'fr': 'fr_fr',
    'de': 'de_de',
    'ru': 'ru_ru',
    'ko': 'ko_kr',
}

for k, lang in LANG_MAPPER.items():
    print(f"Downloading FLEURS {k} ({lang})")
    load_dataset("google/fleurs", lang, split="train")
    load_dataset("google/fleurs", lang, split="test")
