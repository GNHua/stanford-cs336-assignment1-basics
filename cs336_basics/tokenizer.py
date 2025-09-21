from __future__ import annotations

from typing import Iterable, Iterator

import regex as re

from .model import Token, TokenPair, Word

SPLITTER: str = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
SPLITTER_PATTERN = re.compile(SPLITTER)


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.special_tokens_set = set(self.special_tokens or [])

        escaped = sorted([re.escape(t) for t in self.special_tokens], key=len, reverse=True)
        self.special_tokens_pattern = re.compile(f"({'|'.join(escaped)})")

        # self.vocab_tokens = {Token(id, value) for id, value in vocab.items()}
        self.merges_tokens = {
            TokenPair(Token(self.inv_vocab[b1], b1), Token(self.inv_vocab[b2], b2)): i
            for i, (b1, b2) in enumerate(merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab = {}
        with open(vocab_filepath, "rb") as f:
            for line in f:
                k, v = line.strip().split(b"\t", 1)
                vocab[int(k.decode())] = v

        with open(merges_filepath, "rb") as f:
            merges = [tuple(line.strip().split(b"\t", 1)) for line in f]

        return cls(vocab, merges, special_tokens)

    def _pretokenize(self, text: str) -> list[Word]:
        words = []

        text_parts = [text]
        if self.special_tokens:
            text_parts = self.special_tokens_pattern.split(text)

        for part in text_parts:
            if not part:
                continue

            if part.encode() in self.inv_vocab:
                part_bytes = part.encode()
                words.append(Word((Token(self.inv_vocab[part_bytes], part_bytes),)))
                continue

            p_iter = SPLITTER_PATTERN.finditer(part)
            for match in p_iter:
                tokens = []
                word_bytes = match.group().encode()
                if word_bytes in self.inv_vocab:
                    word = Word((Token(self.inv_vocab[word_bytes], word_bytes),))
                else:
                    for b in match.group().encode():
                        bs = bytes([b])
                        token_id = self.inv_vocab[bytes([b])]
                        token = Token(token_id, bs)
                        tokens.append(token)

                    word = Word(tuple(tokens))

                words.append(word)

        return words

    def _encode_word(self, word: Word) -> Word:
        tokens = list(word.tokens)
        while len(tokens) > 1:
            possible_merges = []
            for i in range(len(tokens)-1):
                token_pair = TokenPair(tokens[i], tokens[i + 1])
                if token_pair in self.merges_tokens:
                    possible_merges.append((self.merges_tokens[token_pair], i))

            if not possible_merges:
                break

            best_rank, pair_idx = min(possible_merges)
            new_token_bytes = tokens[pair_idx].value + tokens[pair_idx + 1].value
            new_token = Token(self.inv_vocab[new_token_bytes], new_token_bytes)
            tokens = tokens[:pair_idx] + [new_token] + tokens[pair_idx+2:]

        return Word(tuple(tokens))

    def encode(self, text: str) -> list[int]:
        words = self._pretokenize(text)
        encoded = []

        # key is before and value is after encoding
        cache: dict[Word, Word] = {}

        for word in words:
            if word not in cache:
                cache[word] = self._encode_word(word)
            encoded.extend([t.id for t in cache[word].tokens])

        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: Iterable[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")
