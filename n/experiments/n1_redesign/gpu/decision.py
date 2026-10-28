"""Apply an explicit exclusion decision before the independent sealed-input audit."""
from pathlib import Path


class Decision:
    def __init__(self, rule, corpus, inventory):
        self.folders = set(rule['folders'])
        self.groups = set(rule['corpus_groups'])
        selected = [r for r in corpus if str(r['group_id']) in self.groups or r['set_dir'] in self.folders]
        self.audio = {r['audio_sha256'] for r in selected if r.get('audio_sha256')}
        self.audio |= {r['sha256'] for r in inventory if r.get('sha256') and
                       str(Path(r.get('path', '')).parent) in self.folders}
        self.names = {(r['set_dir'], r['audio_filename'].casefold()) for r in selected}

    def excludes(self, row):
        p = Path(row['audio_rel'])
        return (str(p.parent) in self.folders or row['audio_sha256'] in self.audio or
                (str(p.parent), p.name.casefold()) in self.names or
                bool(set(row.get('groups', [])) & self.groups) or
                str(row.get('group_id', '')) in self.groups)

    def filter(self, rows):
        kept, removed = [], []
        for row in rows:
            (removed if self.excludes(row) else kept).append(row)
        return kept, removed
