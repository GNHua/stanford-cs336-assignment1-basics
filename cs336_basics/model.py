from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import pairwise
from typing import NamedTuple


class Token(NamedTuple):
    id: int
    value: bytes
    merge: TokenPair | None = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Token) -> bool:
        return isinstance(other, Token) and self.id == other.id

    def __len__(self):
        return len(self.value)

    def __lt__(self, other: Token) -> bool:
        return (self.value, len(self.value)) > (other.value, len(other.value))

    def __str__(self):
        return f"Token({self.id}, {self.value})"


class TokenPair(NamedTuple):
    first: Token
    second: Token

    def to_token(self, id: int) -> Token:
        return Token(id, self.first.value + self.second.value, self)

    def to_bytes(self) -> bytes:
        return self.first.value + self.second.value

    def __str__(self):
        return f"({self.first}, {self.second})"


class Word(NamedTuple):
    tokens: tuple[Token, ...]

    @staticmethod
    def from_bytes(byte_data: bytes) -> Word:
        return Word(tuple([Token(b, bytes([b])) for b in byte_data]))

    def merge_tokens(self, token: Token) -> Word:
        assert token.merge

        _tokens = []
        i = 0
        n = len(self.tokens)

        while i < n:
            if (
                i < n - 1
                and self.tokens[i] == token.merge.first
                and self.tokens[i + 1] == token.merge.second
            ):
                _tokens.append(token)
                i += 2
            else:
                _tokens.append(self.tokens[i])
                i += 1

        return Word(tuple(_tokens))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def __str__(self):
        return f"Word({','.join([str(t) for t in self.tokens])})"


@dataclass
class State:
    word_counts: Counter[Word]
    vocab: set[Token] = field(default_factory=set)

    _pair_to_words: defaultdict[TokenPair, set[Word]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _pair_counts: Counter[TokenPair] = field(default_factory=Counter)
    _heap: list[tuple[int, TokenPair]] = field(default_factory=list)

    def __post_init__(self):
        for i in range(256):
            self.vocab.add(Token(i, bytes([i])))

        for word, count in self.word_counts.items():
            for t1, t2 in pairwise(word.tokens):
                token_pair = TokenPair(t1, t2)
                self._pair_to_words[token_pair].add(word)
                self._pair_counts[token_pair] += count

        for token_pair, count in self._pair_counts.items():
            heappush(self._heap, (-count, token_pair))

    def get_best_token_pair(self) -> TokenPair | None:
        while self._heap:
            count, token_pair = -self._heap[0][0], self._heap[0][1]
            if count > 0 and count == self._pair_counts[token_pair]:
                return token_pair
            heappop(self._heap)

    def add_token_to_vocab(self, obj: bytes | Token | TokenPair) -> Token:
        if isinstance(obj, bytes):
            new_token = Token(len(self.vocab), obj)
        elif isinstance(obj, Token):
            new_token = obj
        elif isinstance(obj, TokenPair):
            new_token = obj.to_token(len(self.vocab))

        self.vocab.add(new_token)
        return new_token

    def merge_token_pair(self, token_pair: TokenPair) -> None:
        new_token = self.add_token_to_vocab(token_pair)
        affected_words = self._pair_to_words[token_pair].copy()

        for old_word in affected_words:
            new_word = old_word.merge_tokens(new_token)
            word_count = self.word_counts[old_word]

            del self.word_counts[old_word]
            self.word_counts[new_word] = word_count

            for _pair, count in Counter(pairwise(old_word.tokens)).items():
                pair = TokenPair(*_pair)
                self._pair_to_words[pair].remove(old_word)
                self._pair_counts[pair] -= word_count * count
                heappush(self._heap, (-self._pair_counts[pair], pair))

            for _pair, count in Counter(pairwise(new_word.tokens)).items():
                pair = TokenPair(*_pair)
                self._pair_to_words[pair].add(new_word)
                self._pair_counts[pair] += word_count * count
                heappush(self._heap, (-self._pair_counts[pair], pair))

        del self._pair_to_words[token_pair]
        del self._pair_counts[token_pair]
