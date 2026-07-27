"""Genera i fixture CI del tokenizer K2 (nessun download di modello):

  tests/tok_k2_tiny.json   -- BPE byte-level MINUSCOLO in formato tokenizer.json,
                              prodotto dal VERO convertitore
                              (tools/tiktoken_to_tokenizer_json.py) a partire da
                              un tiktoken.model sintetico addestrato qui;
  tests/tok_k2_cases.txt   -- "testo<TAB>id,id,..." con expected generati da
                              tiktoken.Encoding con la pat_str K2 verbatim
                              sugli stessi rank: l'oracolo e' la stessa engine
                              (fancy-regex) che Kimi usa in produzione.

Il training e' un BPE genuino (256 byte + merge per frequenza, tie-break
deterministico) cosi' la ricostruzione dei merge nel convertitore e' esercitata
per davvero. Eseguire da c/:  python3 tools/make_tok_k2_fixture.py
(richiede: pip install tiktoken)
"""
import base64
import collections
import json
import os
import subprocess
import sys
import tempfile

import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(HERE)

CORPUS = """
hello world Hello World HelloWorld the quick brown fox jumps over lazy dog
dog's don't we'll I'm you're they've he'd it's test testing tested
XMLHttpRequest parseJSON iPhone eBook McDonald
你好 世界 你好世界 中文 汉字 漢字 佐々木 日本語 テキスト
你好嗎 我很好 中文English 混合 mixed text
one two three 123 456 789 1234 12345 version v1
foo bar foo/bar path/to/file print return if else pass
αβγ привет 한국어 ひらがな カタカナ café naïve
""" * 4

N_MERGES = 160


def train_ranks():
    """BPE per frequenza su pezzi separati da whitespace: rank 0-255 = byte,
    poi un merge per iterazione. Deterministico (tie-break sulla coppia)."""
    ranks = {bytes([b]): b for b in range(256)}
    pieces = collections.Counter(
        tuple(bytes([b]) for b in w.encode("utf-8")) for w in CORPUS.split())
    for it in range(N_MERGES):
        pairs = collections.Counter()
        for sym, cnt in pieces.items():
            for i in range(len(sym) - 1):
                pairs[(sym[i], sym[i + 1])] += cnt
        if not pairs:
            break
        (l, r), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        tok = l + r
        ranks[tok] = 256 + it
        newp = collections.Counter()
        for sym, cnt in pieces.items():
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == l and sym[i + 1] == r:
                    out.append(tok); i += 2
                else:
                    out.append(sym[i]); i += 1
            newp[tuple(out)] += cnt
        pieces = newp
    return ranks


def main():
    ranks = train_ranks()
    nbase = len(ranks)

    with tempfile.TemporaryDirectory() as td:
        model = os.path.join(td, "tiktoken.model")
        with open(model, "wb") as f:
            for tok, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
                f.write(base64.b64encode(tok) + b" " + str(rank).encode() + b"\n")
        cfgp = os.path.join(td, "config.json")
        json.dump({"added_tokens_decoder": {
            str(nbase):     {"content": "<|im_end|>"},
            str(nbase + 1): {"content": "[EOS]"},
        }}, open(cfgp, "w"))
        out = os.path.join(C, "tests", "tok_k2_tiny.json")
        subprocess.run([sys.executable,
                        os.path.join(HERE, "tiktoken_to_tokenizer_json.py"),
                        model, out, "--config", cfgp], check=True)

    # pat_str K2 verbatim (tokenization_kimi.py)
    pat = "|".join([
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ])
    specials = {"<|im_end|>": nbase, "[EOS]": nbase + 1}
    for i in range(2, 256):
        specials[f"<|reserved_token_{nbase + i}|>"] = nbase + i
    enc = tiktoken.Encoding(name="k2tiny", pat_str=pat,
                            mergeable_ranks=ranks, special_tokens=specials)

    cases = [
        "hello world", "Hello World", "HelloWorld", "XMLHttpRequest",
        "helloWORLDhello", "dog's don't we'll", "it'S dOn'T", "testing tested",
        "你好世界", "你好, world", "Hello你好World", "中文English中文",
        "汉字漢字", "佐々木", "々", "test々test", "〇一二三",
        "1", "123", "1234", "12345", "v1.2.3", "版本12345号",
        "foo/bar", "a//b", "...", "#!/bin/sh", "end.\n", "-->\n\nnext",
        "  x", "x  ", "a  b", "\t\tz", "\n\n  foo\n", "a\r\nb", "   ",
        "<|im_end|>test[EOS]", "a<|im_end|>b", "你好<|im_end|>世界",
        "αβγ ΑΒΓ", "привет", "한국어", "ひらがなカタカナ", "café naïve",
        "e\u0301clair", "🙂ok", "€100", "print('x')\n\treturn",
        "mixedText123你好'sEnd",
    ]

    def esc(s):
        return (s.replace("\\", "\\\\").replace("\n", "\\n")
                 .replace("\t", "\\t").replace("\r", "\\r"))

    with open(os.path.join(C, "tests", "tok_k2_cases.txt"), "w",
              encoding="utf-8") as f:
        for c in cases:
            ids = enc.encode(c, allowed_special="all")
            f.write(esc(c) + "\t" + ",".join(map(str, ids)) + "\n")
    print(f"fixture: {nbase} base ranks; {len(cases)} cases")


if __name__ == "__main__":
    main()
