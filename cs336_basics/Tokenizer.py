"""Byte-level BPE with ordered merges and protected special tokens."""
import json
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.ids = {v:k for k,v in self.vocab.items()}
        for token in self.special_tokens:
            encoded = token.encode('utf-8')
            if not encoded:
                raise ValueError('Special tokens must not be empty')
            if encoded not in self.ids:
                idx = max(self.vocab, default=-1)+1
                self.vocab[idx], self.ids[encoded] = encoded, idx
        self.ranks = {pair:i for i,pair in enumerate(self.merges)}
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self.pattern = '('+'|'.join(re.escape(s) for s in ordered)+')' if ordered else None
        self.special_set = set(ordered)

    def _piece(self, text):
        pieces = [bytes([b]) for b in text.encode('utf-8')]
        while len(pieces) > 1:
            candidates = [(self.ranks.get((pieces[i],pieces[i+1]), float('inf')), i)
                          for i in range(len(pieces)-1)]
            rank, index = min(candidates)
            if rank == float('inf'):
                break
            pair = (pieces[index], pieces[index+1])
            merged, i = [], 0
            while i < len(pieces):
                if i+1 < len(pieces) and (pieces[i],pieces[i+1]) == pair:
                    merged.append(pieces[i]+pieces[i+1]); i += 2
                else:
                    merged.append(pieces[i]); i += 1
            pieces = merged
        return [self.ids[piece] for piece in pieces]

    def encode(self, text):
        tokens=[]
        for part in re.split(self.pattern,text) if self.pattern else [text]:
            if part in self.special_set:
                tokens.append(self.ids[part.encode('utf-8')])
            else:
                for match in re.finditer(PAT,part):
                    tokens.extend(self._piece(match.group()))
        return tokens

    def encode_iterable(self, iterable):
        # Each iterable item is a text record; callers should use line/document boundaries.
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8',errors='replace')

    def save(self, path):
        payload={'vocab':{str(k):v.hex() for k,v in self.vocab.items()},
                 'merges':[[a.hex(),b.hex()] for a,b in self.merges], 'special_tokens':self.special_tokens}
        with open(path,'w',encoding='utf-8') as f: json.dump(payload,f)

    @classmethod
    def load(cls,path):
        with open(path,encoding='utf-8') as f: data=json.load(f)
        return cls({int(k):bytes.fromhex(v) for k,v in data['vocab'].items()},
                   [(bytes.fromhex(a),bytes.fromhex(b)) for a,b in data['merges']],data['special_tokens'])

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        # GPT-2 JSON vocabulary and merges.txt, using its byte-to-Unicode convention.
        # Build mapping directly to avoid relying on external tokenizer packages.
        visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
        mapping={chr(b):b for b in visible}
        n=0
        for b in range(256):
            if b not in visible:
                mapping[chr(256+n)]=b; n+=1
        def decode(s): return bytes(mapping[c] for c in s)
        with open(vocab_filepath,encoding='utf-8') as f: vocab={i:decode(s) for s,i in json.load(f).items()}
        with open(merges_filepath,encoding='utf-8') as f:
            merges=[tuple(decode(s) for s in line.rstrip().split(' ')) for line in f
                    if not line.startswith('#') and len(line.rstrip().split(' '))==2]
        return cls(vocab,merges,special_tokens)
