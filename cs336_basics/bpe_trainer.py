from collections import Counter, defaultdict
from heapq import heappush
from itertools import pairwise

from .model import State, Token, TokenPair, Word
from .pre_tokenizer import PreTokenizer
import time


class BPETrainer:

    def __init__(self) -> None:
        self.state: State | None = None
        self.merges: list[TokenPair] = []

    def train(
        self, input_path: str, vocab_size: int = 1000, special_tokens: list[str] = []
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        t0 = time.time()

        pre_tokenizer = PreTokenizer(special_tokens=special_tokens, num_processes=4)
        word_counts = pre_tokenizer.tokenize(input_path)

        t1 = time.time()

        self.state = State(word_counts)

        for special_token in special_tokens:
            self.state.add_token_to_vocab(special_token.encode("utf-8"))

        while len(self.state.vocab) < vocab_size:
            best_token_pair = self.state.get_best_token_pair()
            if not best_token_pair:
                break

            self.state.merge_token_pair(best_token_pair)

        t2 = time.time()

        sorted_tokens = sorted(self.state.vocab, key=lambda x: x.id)
        final_vocab = {token.id: token.value for token in sorted_tokens}
        final_merges = [
            (t.merge.first.value, t.merge.second.value)
            for t in sorted_tokens
            if t.merge
        ]

        print(f"\n>>>>>>>>>>>>>>>>> pre tokenize: {t1-t0:.2f}s")
        print(f">>>>>>>>>>>>>>>>> train: {t2-t1:.2f}s")

        return final_vocab, final_merges
