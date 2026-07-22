"""Embed Hanja candidates into POS-tagged Korean sentences.

Consumes the surface/POS format from tokenize_data.py, identifies
Sino-Korean candidate words (nouns), and inserts Hanja candidates
after them. Strips POS tags from the output.
"""
from preprocessing.tokenize import POS_SEP

# POS tags that can be Sino-Korean (mecab-ko-dic tagset)
# NNG: common noun, NNP: proper noun
SK_CANDIDATE_POS = {"NNG", "NNP"}


def append_hanja(tagged_sentence: str, lookup: dict) -> str:
    """Insert Hanja candidates after SK nouns in a POS-tagged sentence.

    Input:  "나/NP 는/JX 사과/NNG 의/JKG 가격/NNG 을/JKO"
    Output: "나 는 사과 沙果 謝過 의 가격 價格 加擊 을"
    """
    result = []
    # Tokens are always separated by spaces, and the last '/' in each
    # token always identifies the part of speech (POS) tag.
    for token in tagged_sentence.split():
        parts = token.rsplit(POS_SEP, 1)
        if len(parts) != 2:
            result.append(token)
            continue
        surface, pos = parts
        result.append(surface)
        if pos in SK_CANDIDATE_POS:
            candidates = lookup.get(surface, [])
            result.extend(candidates)
    return " ".join(result)

