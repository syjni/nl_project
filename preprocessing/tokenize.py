"""
MeCab tokenization for Japanese and Korean.

Uses a single MeCab engine (pip: mecab) with two dictionaries:
  - ipadic for Japanese
  - mecab-ko-dic for Korean

Korean output retains part of speech (POS) tags in surface/POS format for downstream use
(e.g. Hanja embedding). Japanese output is plain tokens (no part of speech needed).
"""
import os
# The pip mecab package looks for a config file at /usr/local/etc/mecabrc on startup.
# It doesn't need anything from it, but fails if the file doesn't exist.
# Pointing it to /dev/null satisfies the check.
os.environ.setdefault("MECABRC", "/dev/null")

import MeCab
import ipadic
import mecab_ko_dic

import opencc

POS_SEP = "/"  # Separator between surface and part of speech (POS) tag

_ja_tagger = MeCab.Tagger(f"-Owakati -d {ipadic.DICDIR}")  # wakati is japanese (分かち) meaning space-separated tokens
_ko_tagger = MeCab.Tagger(f"-d {mecab_ko_dic.DICDIR}")
_jp2t = opencc.OpenCC("jp2t.json")


def tokenize_ja(text):
    """Tokenize Japanese text, normalize Kanji to Traditional Chinese."""
    # Tokenize text
    tokens = _ja_tagger.parse(text).strip()

    # Normalize kanji to traditional chinese.
    return _jp2t.convert(tokens)


def tokenize_ko(text):
    """Tokenize Korean text, returning surface/POS (part of speech) pairs.

    Example: "사과의 가격" -> "사과/NNG 의/JKG 가격/NNG"

    Read about POS tags: https://docs.google.com/spreadsheets/d/1-9blXKjtjeKZqsf4NzHeYJCrr49-nXeRF6D80udfcwY
    """
    node = _ko_tagger.parseToNode(text)
    tokens = []
    while node:
        if node.surface:
            pos = node.feature.split(",")[0]
            tokens.append(f"{node.surface}{POS_SEP}{pos}")
        node = node.next
    return " ".join(tokens)


def strip_pos(tagged_sentence):
    """Remove POS tags from a tagged sentence: 'surface/POS ...' -> 'surface ...'"""
    return " ".join(tok.rsplit(POS_SEP, 1)[0] for tok in tagged_sentence.split())
