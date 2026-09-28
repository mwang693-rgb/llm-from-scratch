"""BPE training with local pair-count updates and deterministic tie breaking."""
from collections import Counter, defaultdict
import regex as re
from .Tokenizer import PAT

def training_bpe(input_path, vocab_size, special_tokens):
    special_tokens=list(dict.fromkeys(special_tokens))
    vocab={i:bytes([i]) for i in range(256)}
    for s in special_tokens:
        if not s: raise ValueError('Special tokens must not be empty')
        vocab[len(vocab)] = s.encode('utf-8')
    if vocab_size < len(vocab): raise ValueError('Vocabulary too small for bytes and special tokens')
    with open(input_path,encoding='utf-8') as f: text=f.read()
    pattern='|'.join(re.escape(s) for s in sorted(special_tokens,key=len,reverse=True))
    words=Counter()
    for part in re.split(pattern,text) if pattern else [text]:
        words.update(m.group().encode('utf-8') for m in re.finditer(PAT,part))
    sequences={i:tuple(bytes([b]) for b in w) for i,w in enumerate(words)}
    frequencies=list(words.values())
    counts=Counter()
    owners=defaultdict(set)
    for i,word in sequences.items():
        for pair,n in Counter(zip(word,word[1:])).items():
            counts[pair] += n*frequencies[i]; owners[pair].add(i)
    merges=[]
    while len(vocab)<vocab_size and counts:
        pair=max(counts,key=lambda p:(counts[p],p))
        merges.append(pair)
        vocab[len(vocab)]=b''.join(pair)
        for idx in list(owners[pair]):
            old=sequences[idx]
            for p,n in Counter(zip(old,old[1:])).items():
                counts[p] -= n*frequencies[idx]
                owners[p].discard(idx)
                if counts[p]==0: del counts[p]
            new=[]; j=0
            while j<len(old):
                if j+1<len(old) and (old[j],old[j+1])==pair:
                    new.append(old[j]+old[j+1]); j+=2
                else:
                    new.append(old[j]); j+=1
            new=tuple(new); sequences[idx]=new
            for p,n in Counter(zip(new,new[1:])).items():
                counts[p] += n*frequencies[idx]; owners[p].add(idx)
    return vocab,merges
