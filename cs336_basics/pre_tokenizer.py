import os
from collections import Counter
from itertools import pairwise
from multiprocessing import Pool
from typing import BinaryIO

import regex as re
from .model import Word


class PreTokenizer:

    def __init__(
        self,
        chunk_boundary_token: bytes = b"<|endoftext|>",
        special_tokens: list[str] = ["<|endoftext|>"],
        splitter: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        num_processes: int | None = None,
    ) -> None:
        self.chunk_boundary_token = chunk_boundary_token
        self.special_tokens = special_tokens
        self.splitter_pattern = re.compile(splitter)
        self.num_processes = num_processes or os.cpu_count() or 1

    def tokenize(self, path: str) -> Counter[Word]:
        with open(path, "rb") as f:
            boundaries = PreTokenizer.find_chunk_boundaries(
                f, self.num_processes, self.chunk_boundary_token
            )

        with Pool(processes=self.num_processes) as pool:
            list_of_args = [
                (path, start, end, self.splitter_pattern, self.special_tokens)
                for start, end in pairwise(boundaries)
            ]
            list_of_result_dicts = pool.starmap(
                PreTokenizer.process_chunk, list_of_args
            )

        word_counts: Counter[Word] = Counter()
        for d in list_of_result_dicts:
            for word, count in d.items():
                word_counts[word] += count

        return word_counts

    @staticmethod
    def process_chunk(
        path: str,
        start: int,
        end: int,
        splitter_pattern: re.Pattern,
        special_tokens: list[str],
    ) -> Counter[Word]:
        local_word_counts: Counter[Word] = Counter()

        # TODO: Refactor - Inefficient on large files
        with open(path, "rb") as f:
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="replace")

            text_parts = [chunk_text]
            for token in special_tokens:
                # For each special token, split every existing part further
                new_parts = []
                for part in text_parts:
                    new_parts.extend(part.split(token))
                text_parts = new_parts

            # Now text_parts contains only the text BETWEEN special tokens
            for sub_chunk in text_parts:
                if not sub_chunk:
                    continue
                p_iter = splitter_pattern.finditer(sub_chunk)
                for match in p_iter:
                    word = Word.from_bytes(match.group().encode("utf-8"))
                    local_word_counts[word] += 1

        return local_word_counts

    @staticmethod
    def find_chunk_boundaries(
        file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
