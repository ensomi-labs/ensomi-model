"""Keysound lexicon copied from supervision_names.py at d9b366a."""
import os
import re

PARTS = {                                   # unambiguous tokens only: no two-letter channel-like ones (ch, oh, se, ...)
    'kick': 'kick', 'bd': 'kick', 'bassdrum': 'kick', 'snare': 'snare', 'sd': 'snare',
    'clap': 'clap', 'hat': 'hat', 'hh': 'hat', 'hihat': 'hat', 'cymbal': 'cymbal',
    'crash': 'cymbal', 'cym': 'cymbal', 'ride': 'cymbal', 'splash': 'cymbal', 'china': 'cymbal', 'tom': 'tom',
    'perc': 'perc', 'shaker': 'perc', 'tamb': 'perc', 'drum': 'drum', 'drums': 'drum',
    'bass': 'bass', 'sub': 'bass', 'piano': 'keys', 'pf': 'keys', 'pno': 'keys',
    'keys': 'keys', 'organ': 'keys', 'synth': 'synth', 'syn': 'synth', 'lead': 'synth',
    'pad': 'synth', 'arp': 'synth', 'pluck': 'synth', 'saw': 'synth', 'guitar': 'guitar', 'gt': 'guitar',
    'gtr': 'guitar', 'string': 'strings', 'strings': 'strings', 'str': 'strings', 'violin': 'strings',
    'vocal': 'vocal', 'vox': 'vocal', 'voice': 'vocal', 'chorus': 'vocal', 'vocals': 'vocal',
    'bell': 'bell', 'bells': 'bell', 'brass': 'brass', 'trumpet': 'brass', 'sax': 'brass', 'flute': 'wind',
    'fx': 'fx', 'sfx': 'fx', 'noise': 'fx', 'riser': 'fx', 'chord': 'chord', 'chords': 'chord',
}

TOKEN = re.compile(r'[a-z]+|\d+')

def tokens(name: str) -> list[str]:
    stem = os.path.splitext(os.path.basename(name.replace('\\', '/')))[0].casefold()
    return TOKEN.findall(stem)

def part_of(name: str) -> str | None:
    for t in tokens(name):
        if t in PARTS:
            return PARTS[t]
    return None
