"""Tokenization helpers for silence-token timelines.

Predictions and references are always **token strings** — text with the
silence/wait markers embedded (e.g. ``"<silence> stew <|im_end|>"``).  This
is the uniform interface for every model and API; convert other formats
(HF token ids, decoded strings without markers) in a preprocessing step
before evaluation.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Set, Union

import torch


class VocabTokenizer:
    """Minimal id↔string tokenizer used when sequences are already tokenized."""

    def __init__(
        self,
        silence_token: str,
        eos_tokens: Optional[Sequence[str]] = None,
        extra_tokens: Optional[Iterable[str]] = None,
    ):
        self.silence_token = silence_token
        self.eos_tokens: Set[str] = set(eos_tokens or ())
        self.special_tokens: Set[str] = {silence_token} | self.eos_tokens
        self.token_to_id = {silence_token: 0}
        self.id_to_token = {0: silence_token}
        if extra_tokens:
            for tok in extra_tokens:
                self.add_token(tok)

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    @property
    def silence_token_id(self) -> int:
        return self.token_to_id[self.silence_token]

    def encode_tokens(self, tokens: Sequence[str]) -> torch.Tensor:
        ids = [self.add_token(t) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        pieces = []
        for i in ids:
            tok = self.id_to_token.get(int(i), "")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            pieces.append(tok)
        # Prefer space-join for word-like tokens; keep BPE pieces glued if they
        # look like continuation pieces (leading ▁/Ġ stripped elsewhere).
        return " ".join(p for p in pieces if p)


def split_special_token_string(text: str, special_tokens: Sequence[str]) -> List[str]:
    """Split a decoded timeline string into tokens using known special tokens.

    Special tokens are matched literally; remaining spans are whitespace-split
    into content tokens. Useful when predictions are dumped as one string.
    """
    if not text:
        return []
    # Longest-first so overlapping names resolve correctly.
    specials = sorted(set(special_tokens), key=len, reverse=True)
    if not specials:
        return text.split()
    pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
    parts = re.split(pattern, text)
    tokens: List[str] = []
    special_set = set(specials)
    for part in parts:
        if not part:
            continue
        if part in special_set:
            tokens.append(part)
        else:
            tokens.extend(part.split())
    return tokens


def normalize_sequence(
    seq: Union[str, Sequence[str], Sequence[int]],
    *,
    silence_token: str,
    eos_tokens: Sequence[str],
) -> List[str]:
    """Normalize a JSONL ``pred``/``ref`` field into a list of token strings."""
    if isinstance(seq, str):
        specials = [silence_token, *eos_tokens]
        return split_special_token_string(seq, specials)

    if len(seq) == 0:
        return []

    if isinstance(seq[0], int):
        raise ValueError(
            "Integer token-id sequences are not supported: convert them to "
            "token strings (the uniform text-with-silence-marks interface) "
            "in a preprocessing step."
        )

    return [str(t) for t in seq]
