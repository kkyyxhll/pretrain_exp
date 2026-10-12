"""Pack original document tokens into resumable masked training batches."""

from __future__ import annotations

import copy
import os
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from src.data.corpus import CorpusReader, CorpusState, Document, RecordPolicy
from src.data.tokenization import MAX_ENCODING_THREADS, encode_threads

IGNORE_INDEX = -100
DEFAULT_DOCUMENT_BATCH_SIZE = 128


class ReaderCounters(TypedDict):
    """Keep counters aligned with the restored corpus cursor."""

    documents_yielded: int
    lines_blank: int
    bad_utf8: int


class PackedState(TypedDict, total=False):
    """Describe the established version-two and version-three checkpoint fields."""

    format_version: Literal[2, 3]
    corpus: CorpusState | None
    skip_tokens: int
    batch_size: int
    sequence_length: int
    mask_separator: bool
    documents_seen: int
    tokens_produced: int
    batches_produced: int
    loop: bool
    drop_last: bool
    doc_batch: int
    next_state: CorpusState | None
    tokens: NDArray[np.int32]
    is_text: NDArray[np.bool_]
    docs: list[list[CorpusState | int | None]]
    exhausted: bool
    reader_counters: ReaderCounters


class StreamingTokenizer(Protocol):
    """Expose the encoding and byte metadata required by the document packer."""

    bos_id: int | None
    eos_id: int
    pad_id: int
    byte_lengths: NDArray[np.int64]

    def encode_batch(self, texts: Sequence[str], threads: int | None = None) -> list[list[int]]:
        """Encode original documents without adding separator tokens."""
        ...


class DataLoaderError(RuntimeError):
    """Report invalid packing settings or incompatible stream checkpoints."""


@dataclass(kw_only=True)
class Batch:
    """Hold packed token rows and the established accounting fields."""

    inputs: NDArray[np.int64]
    labels: NDArray[np.int64]
    scored_tokens: int
    scored_bytes: int
    padded_positions: int
    state: PackedState = field(default_factory=PackedState)


class PackedDataLoader:
    """Yield one rank's packed batches with an exact finite prefetch checkpoint."""

    def __init__(
        self,
        tokenizer: StreamingTokenizer,
        paths: Sequence[str | Path],
        *,
        batch_size: int,
        sequence_length: int,
        rank: int = 0,
        world_size: int = 1,
        loop: bool = True,
        drop_last: bool = True,
        doc_batch: int = DEFAULT_DOCUMENT_BATCH_SIZE,
        threads: int | None = None,
        on_bad_utf8: RecordPolicy = "skip",
        mask_separator: bool = True,
        on_blank: RecordPolicy = "skip",
    ) -> None:
        for setting_name, setting_value in (
            ("batch_size", batch_size),
            ("sequence_length", sequence_length),
            ("doc_batch", doc_batch),
        ):
            if (
                not isinstance(setting_value, int)
                or isinstance(setting_value, bool)
                or setting_value < 1
            ):
                raise DataLoaderError(
                    f"{setting_name} must be a positive int, got {setting_value!r}"
                )
            else:
                pass
        self.tokenizer: StreamingTokenizer = tokenizer
        self.batch_size: int = batch_size
        self.sequence_length: int = sequence_length
        self.loop: bool = loop
        self.drop_last: bool = drop_last
        self.doc_batch: int = doc_batch
        self.mask_separator: bool = mask_separator
        self.reader: CorpusReader = CorpusReader(
            paths,
            rank=rank,
            world_size=world_size,
            on_bad_utf8=on_bad_utf8,
            on_blank=on_blank,
            loop=loop,
        )
        if threads is None:
            threads = max(
                1,
                min(
                    MAX_ENCODING_THREADS,
                    (os.cpu_count() or MAX_ENCODING_THREADS) // max(1, world_size),
                ),
            )
        else:
            pass
        self.encoding_thread_count: int = encode_threads(threads)
        self.bos_id: int | None = tokenizer.bos_id
        self.eos_id: int = tokenizer.eos_id
        self.pad_id: int = tokenizer.pad_id
        self.token_byte_lengths: NDArray[np.int64] = tokenizer.byte_lengths
        self.buffer_token_ids: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self.buffer_text_mask: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)
        self.prefetched_documents: deque[list[CorpusState | int | None]] = deque()
        self.head_document_token_offset: int = 0
        self.next_corpus_state: CorpusState | None = None
        self.resume_corpus_state: CorpusState | None = None
        self.pending_skip_token_count: int = 0
        self.documents_seen: int = 0
        self.tokens_produced: int = 0
        self.batches_produced: int = 0
        self.is_exhausted: bool = False
        self.document_iterator: Iterator[Document] | None = None

    def state_dict(self) -> PackedState:
        """Save prefetched tokens without replaying or recounting documents."""
        return {
            "format_version": 3,
            "corpus": (
                self.prefetched_documents[0][0]
                if self.prefetched_documents
                else self.next_corpus_state
            ),
            "skip_tokens": self.head_document_token_offset,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "mask_separator": self.mask_separator,
            "documents_seen": self.documents_seen,
            "tokens_produced": self.tokens_produced,
            "batches_produced": self.batches_produced,
            "loop": self.loop,
            "drop_last": self.drop_last,
            "doc_batch": self.doc_batch,
            "next_state": copy.deepcopy(self.next_corpus_state),
            "tokens": self.buffer_token_ids.copy(),
            "is_text": self.buffer_text_mask.copy(),
            "docs": copy.deepcopy(list(self.prefetched_documents)),
            "exhausted": self.is_exhausted,
            "reader_counters": {
                "documents_yielded": self.reader.documents_yielded,
                "lines_blank": self.reader.lines_blank,
                "bad_utf8": self.reader.bad_utf8,
            },
        }

    def load_state_dict(self, state: PackedState) -> None:
        """Restore the same packing configuration and complete prefetch state."""
        state_version = state.get("format_version")
        if state_version not in (2, 3):
            raise DataLoaderError(
                f"resume state format_version {state.get('format_version')!r} is not 2 or 3"
            )
        else:
            packing_settings: dict[str, int | bool] = {
                "sequence_length": self.sequence_length,
                "mask_separator": self.mask_separator,
            }
            if state_version == 3:
                packing_settings.update(
                    batch_size=self.batch_size,
                    loop=self.loop,
                    drop_last=self.drop_last,
                    doc_batch=self.doc_batch,
                )
            else:
                pass
            for setting_name, setting_value in packing_settings.items():
                if state.get(setting_name) != setting_value:
                    raise DataLoaderError(
                        f"resume state has {setting_name}={state.get(setting_name)!r} "
                        f"but this loader has {setting_value!r}; the packing would not line up"
                    )
                else:
                    pass
        self.resume_corpus_state = state["corpus"]
        self.pending_skip_token_count = int(state["skip_tokens"])
        self.buffer_token_ids = np.empty(0, dtype=np.int32)
        self.buffer_text_mask = np.empty(0, dtype=np.bool_)
        self.prefetched_documents.clear()
        self.head_document_token_offset = 0
        self.documents_seen = state.get("documents_seen", 0)
        self.tokens_produced = state.get("tokens_produced", 0)
        self.batches_produced = state.get("batches_produced", 0)
        self.document_iterator = None
        self.next_corpus_state = copy.deepcopy(self.resume_corpus_state)
        self.is_exhausted = False
        if state_version == 3:
            self.buffer_token_ids = np.asarray(state["tokens"], dtype=np.int32).copy()
            self.buffer_text_mask = np.asarray(state["is_text"], dtype=np.bool_).copy()
            if (
                self.buffer_token_ids.ndim != 1
                or self.buffer_token_ids.shape != self.buffer_text_mask.shape
            ):
                raise DataLoaderError("invalid checkpoint token/mask buffer")
            else:
                pass
            self.prefetched_documents = deque(copy.deepcopy(state["docs"]))
            self.head_document_token_offset = int(state["skip_tokens"])
            self.pending_skip_token_count = 0
            self.next_corpus_state = copy.deepcopy(state["next_state"])
            self.resume_corpus_state = copy.deepcopy(self.next_corpus_state)
            self.is_exhausted = state["exhausted"]
            self.reader.documents_yielded = state["reader_counters"]["documents_yielded"]
            self.reader.lines_blank = state["reader_counters"]["lines_blank"]
            self.reader.bad_utf8 = state["reader_counters"]["bad_utf8"]
            if self.next_corpus_state is not None:
                self.reader.position = (
                    self.next_corpus_state["epoch"],
                    self.next_corpus_state["shard"],
                    self.next_corpus_state["line"],
                )
            else:
                pass
        else:
            pass

    def flush_documents(
        self, document_texts: list[str], corpus_states: list[CorpusState | None]
    ) -> None:
        """Encode prefetched original documents and mask only appended separators."""
        if not document_texts:
            return
        else:
            encoded_documents = self.tokenizer.encode_batch(
                document_texts, threads=self.encoding_thread_count
            )
            document_token_counts = [len(token_ids) + 1 for token_ids in encoded_documents]
            total_token_count = sum(document_token_counts)
            token_ids_buffer = np.empty(total_token_count, dtype=np.int32)
            text_mask = np.ones(total_token_count, dtype=np.bool_)
            token_offset = 0
            for token_ids in encoded_documents:
                document_token_count = len(token_ids)
                if document_token_count:
                    token_ids_buffer[token_offset : token_offset + document_token_count] = token_ids
                else:
                    pass
                token_ids_buffer[token_offset + document_token_count] = self.eos_id
                text_mask[token_offset + document_token_count] = False
                token_offset += document_token_count + 1
            self.buffer_token_ids = np.concatenate((self.buffer_token_ids, token_ids_buffer))
            self.buffer_text_mask = np.concatenate((self.buffer_text_mask, text_mask))
            for corpus_state, document_token_count in zip(corpus_states, document_token_counts):
                self.prefetched_documents.append([corpus_state, document_token_count])
            document_texts.clear()
            corpus_states.clear()

    def fill_buffer(self, documents: Iterator[Document], needed_token_count: int) -> None:
        """Prefetch until the token buffer is large enough or the stream ends."""
        document_texts: list[str] = []
        corpus_states: list[CorpusState | None] = []
        while len(self.buffer_token_ids) < needed_token_count and not self.is_exhausted:
            try:
                document = next(documents)
            except StopIteration:
                self.is_exhausted = True
                break
            # note (kkyyxhll): The preceding cursor resumes at the next prefetched document.
            corpus_states.append(self.next_corpus_state)
            document_texts.append(document.text)
            self.next_corpus_state = self.reader.state()
            self.documents_seen += 1
            if len(document_texts) >= self.doc_batch:
                self.flush_documents(document_texts, corpus_states)
            else:
                pass
        self.flush_documents(document_texts, corpus_states)

    def drop_tokens(self, token_count: int) -> None:
        """Advance the buffered stream and retire fully consumed documents."""
        self.buffer_token_ids = self.buffer_token_ids[token_count:]
        self.buffer_text_mask = self.buffer_text_mask[token_count:]
        self.head_document_token_offset += token_count
        while (
            self.prefetched_documents
            and self.head_document_token_offset >= self.prefetched_documents[0][1]
        ):
            self.head_document_token_offset -= self.prefetched_documents[0][1]
            self.prefetched_documents.popleft()

    def token_row(self, token_offset: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Shift targets by one token while keeping adjacent input rows contiguous."""
        input_ids = self.buffer_token_ids[
            token_offset : token_offset + self.sequence_length
        ].astype(np.int64)
        target_slice = slice(token_offset + 1, token_offset + self.sequence_length + 1)
        labels = self.buffer_token_ids[target_slice].astype(np.int64)
        if self.mask_separator:
            labels = np.where(self.buffer_text_mask[target_slice], labels, IGNORE_INDEX)
        else:
            pass
        return input_ids, labels

    def batches(self) -> Iterator[Batch]:
        """Yield complete batches and optionally pad a final partial token row."""
        if self.document_iterator is None:
            self.document_iterator = self.reader.documents(resume=self.resume_corpus_state)
        else:
            pass
        required_row_tokens = self.sequence_length + 1
        if self.pending_skip_token_count:
            self.fill_buffer(
                self.document_iterator, self.pending_skip_token_count + required_row_tokens
            )
            if len(self.buffer_token_ids) < self.pending_skip_token_count:
                raise DataLoaderError(
                    f"resume needs to discard {self.pending_skip_token_count} tokens but the "
                    f"corpus only produced {len(self.buffer_token_ids)} from that position; "
                    "the shard list or tokenizer is not the one that wrote this state"
                )
            else:
                self.drop_tokens(self.pending_skip_token_count)
                self.pending_skip_token_count = 0
        else:
            pass
        while True:
            input_rows: list[NDArray[np.int64]] = []
            label_rows: list[NDArray[np.int64]] = []
            while len(input_rows) < self.batch_size:
                self.fill_buffer(self.document_iterator, required_row_tokens)
                if len(self.buffer_token_ids) >= required_row_tokens:
                    input_ids, labels = self.token_row(0)
                    self.drop_tokens(self.sequence_length)
                elif self.is_exhausted and not self.drop_last and len(self.buffer_token_ids) > 1:
                    available_token_count = len(self.buffer_token_ids) - 1
                    input_ids = np.full(self.sequence_length, self.pad_id, dtype=np.int64)
                    labels = np.full(self.sequence_length, IGNORE_INDEX, dtype=np.int64)
                    partial_inputs, partial_labels = self.token_row(0)
                    input_ids[:available_token_count] = partial_inputs[:available_token_count]
                    labels[:available_token_count] = partial_labels[:available_token_count]
                    self.drop_tokens(len(self.buffer_token_ids))
                else:
                    break
                input_rows.append(input_ids)
                label_rows.append(labels)
            if not input_rows:
                return
            else:
                input_ids = np.stack(input_rows)
                labels = np.stack(label_rows)
                scored_mask = labels != IGNORE_INDEX
                self.tokens_produced += int(input_ids.size)
                self.batches_produced += 1
                yield Batch(
                    inputs=input_ids,
                    labels=labels,
                    scored_tokens=int(scored_mask.sum()),
                    scored_bytes=int(self.token_byte_lengths[labels[scored_mask]].sum()),
                    padded_positions=(
                        int((input_ids == self.pad_id).sum()) if not self.drop_last else 0
                    ),
                    state=self.state_dict(),
                )
            if self.is_exhausted and len(self.buffer_token_ids) <= 1:
                return
            else:
                pass
