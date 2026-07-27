"""tiktoken.model -> tokenizer.json (formato HF caricabile da tok.h).

Kimi K2 spedisce il vocabolario come tiktoken.model (righe "base64(bytes) rank")
piu' tokenization_kimi.py; tok.h invece parla solo tokenizer.json (BPE
byte-level con merges espliciti). Questo tool fa il ponte:

  1. carica i rank tiktoken (load_tiktoken_bpe);
  2. RICOSTRUISCE i merges: per ogni token multi-byte, BPE-codifica il token
     stesso usando solo rank inferiori al suo -- lo stato a un passo dalla fine
     e' la coppia (left,right) che il training BPE fuse per crearlo. E' la
     ricostruzione canonica (stessa usata dalle conversioni HF di o200k/cl100k);
     ogni token da' esattamente un merge, ordinato per rank del risultato, che
     e' esattamente la priorita' con cui tiktoken fonde le coppie in encode.
  3. mappa i byte in stringhe byte-level GPT-2 (bytes_to_unicode) -- la stessa
     mappa hardcoded in tk_build_bytemap();
  4. emette added_tokens dai 256 slot speciali riservati (config o
     <|reserved_token_N|>) e il pre_tokenizer Split con la pat_str K2
     (il fingerprint \\p{Han} che tok.h usa per selezionare la famiglia).

Uso:
  python3 tools/tiktoken_to_tokenizer_json.py tiktoken.model out/tokenizer.json \
      [--config tokenizer_config.json] [--pat k2]

Verifica a valle (sempre): tok_encode C vs tiktoken.Encoding sul corpus
avversariale -- vedi tests/test_tok_k2.c per il contratto CI.
"""
import argparse
import base64
import json
import sys

# La pat_str di tokenization_kimi.py, verbatim (sintassi fancy-regex: le
# classi con && intersezione non sono re/regex standard Python).
K2_PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

NUM_RESERVED = 256          # tokenization_kimi.py: num_reserved_special_tokens


def bytes_to_unicode():
    """GPT-2 byte-level map, identica a tk_build_bytemap() in tok.h."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_ranks(path):
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok_b64, rank = line.split()
            ranks[base64.b64decode(tok_b64)] = int(rank)
    return ranks


def recover_merge(ranks, token, rank):
    """Lo split (left,right) che il training fuse per creare `token`:
    BPE sul token stesso con soli rank < rank; l'ultimo stato a 2 simboli."""
    parts = [bytes([b]) for b in token]
    while len(parts) > 2:
        best_rank, best_i = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < rank and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_i is None:
            return None                      # non ricostruibile (token "orfano")
        parts = parts[:best_i] + [parts[best_i] + parts[best_i + 1]] + parts[best_i + 2:]
    if len(parts) != 2:
        return None
    if parts[0] not in ranks or parts[1] not in ranks:
        return None
    return parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="tiktoken.model (righe 'base64 rank')")
    ap.add_argument("out", help="tokenizer.json di destinazione")
    ap.add_argument("--config", help="tokenizer_config.json (nomi degli special token)")
    ap.add_argument("--pat", default="k2", choices=["k2"],
                    help="famiglia pre-tokenizer (solo k2 per ora)")
    args = ap.parse_args()

    ranks = load_ranks(args.model)
    b2u = bytes_to_unicode()

    def bl(tok: bytes) -> str:
        return "".join(b2u[b] for b in tok)

    vocab = {bl(tok): rank for tok, rank in ranks.items()}
    if len(vocab) != len(ranks):
        sys.exit("byte-level collision: vocab non iniettivo (impossibile)")

    merges, orphans = [], []
    for tok, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(tok) < 2:
            continue
        m = recover_merge(ranks, tok, rank)
        if m is None:
            orphans.append((tok, rank))
            continue
        merges.append([bl(m[0]), bl(m[1])])
    if orphans:
        # Un vocab BPE genuino non ne ha; se compaiono, il modello non e' BPE
        # puro e la parita' con tiktoken NON e' garantita: fallire forte.
        for tok, rank in orphans[:10]:
            print(f"  orphan rank={rank} bytes={tok!r}", file=sys.stderr)
        sys.exit(f"{len(orphans)} token non ricostruibili come merge")

    # Special tokens: 256 slot riservati dopo i rank base (tokenization_kimi.py)
    names = {}
    if args.config:
        cfg = json.load(open(args.config))
        for i, d in cfg.get("added_tokens_decoder", {}).items():
            names[int(i)] = d["content"]
    nbase = len(ranks)
    added = []
    for i in range(nbase, nbase + NUM_RESERVED):
        added.append({
            "id": i,
            "content": names.get(i, f"<|reserved_token_{i}|>"),
            "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": False, "special": True,
        })

    doc = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split",
                 "pattern": {"Regex": K2_PAT},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
            "vocab": vocab, "merges": merges,
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print(f"wrote {args.out}: {len(vocab)} vocab, {len(merges)} merges, "
          f"{len(added)} added tokens")


if __name__ == "__main__":
    main()
