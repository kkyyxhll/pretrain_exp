#!/usr/bin/env python3
"""Streaming dataloader: documents in, packed training batches out.

Packing follows eval/official/eval.py exactly, because a training loss computed
over a differently shaped stream is not the quantity the leaderboard measures:

* every document is terminated with <|eos|> -- the tokenizer's HuggingFace
  eos_token and the same token eval.py:44 writes between documents;
* the per-document streams are concatenated into one flat stream and sliced
  with stride `sequence_length`, so a row needs `sequence_length + 1` tokens and
  consecutive rows share one token at the boundary (eval.py:62-66);
* a target position is scored only where `is_text` is true, so the inserted
  separator is never a prediction target. Matching this means training loss is
  the same quantity as the reported bpb rather than an approximation of it.
  `mask_separator=False` trains on the separator instead, at the cost of that
  correspondence.

Tokenization happens on the fly. Each rank reads its document-level stride over
all compressed shards; no decompressed corpus or token cache is written.

Batches are numpy. Nothing here imports torch or assumes a device, because no
training backend is selected yet; `torch_batch()` is a thin opt-in helper.
"""

import collections
import copy
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .corpus import CorpusReader
from .tokenization import encode_threads

IGNORE_INDEX = -100  # torch's cross-entropy default, and eval.py:60's
DEFAULT_DOC_BATCH = 128  # eval.py:34 tokenizes 128 documents at a time


class DataLoaderError(RuntimeError):
    """A batching or configuration problem the caller must resolve."""


def bits_per_byte(nll_nats, source_bytes) -> float:
    """eval/official/eval.py:225's metric: summed NLL in nats over source bytes."""
    if source_bytes <= 0:
        raise DataLoaderError(f"bits per byte needs a positive byte count, got {source_bytes}")
    return nll_nats / (math.log(2) * source_bytes)


@dataclass
class Batch:
    """One packed batch, plus the accounting a comparable bpb needs."""

    inputs: np.ndarray  # (rows, sequence_length) int64
    labels: np.ndarray  # (rows, sequence_length) int64, IGNORE_INDEX elsewhere
    scored_tokens: int  # label positions that are not IGNORE_INDEX
    scored_bytes: int  # source UTF-8 bytes behind those positions
    padded_positions: int  # tail padding, only possible when drop_last=False
    state: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return int(self.inputs.shape[0])

    def bits_per_byte(self, nll_nats) -> float:
        """Convert this batch's summed NLL into the official metric."""
        return bits_per_byte(nll_nats, self.scored_bytes)


def torch_batch(batch, *, device=None, pin=False):
    """Convert to torch tensors. Opt-in, so the data path stays torch-free."""
    import torch

    inputs = torch.from_numpy(batch.inputs)
    labels = torch.from_numpy(batch.labels)
    if pin:
        inputs, labels = inputs.pin_memory(), labels.pin_memory()
    if device is not None:
        inputs = inputs.to(device, non_blocking=pin)
        labels = labels.to(device, non_blocking=pin)
    return inputs, labels


class PackedDataLoader:
    """Yields packed batches for one rank, forever by default."""

    def __init__(
        self,
        tokenizer,
        paths,
        *,
        batch_size,
        sequence_length,
        rank=0,
        world_size=1,
        loop=True,
        drop_last=True,
        doc_batch=DEFAULT_DOC_BATCH,
        threads=None,
        on_bad_utf8="skip",
        mask_separator=True,
        on_blank="skip",
    ):
        for name, value in (
            ("batch_size", batch_size),
            ("sequence_length", sequence_length),
            ("doc_batch", doc_batch),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise DataLoaderError(f"{name} must be a positive int, got {value!r}")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.loop = loop
        self.drop_last = drop_last
        self.doc_batch = doc_batch
        self.mask_separator = mask_separator
        self.reader = CorpusReader(
            paths,
            rank=rank,
            world_size=world_size,
            on_bad_utf8=on_bad_utf8,
            on_blank=on_blank,
            loop=loop,
        )
        # Ranks share a node, so divide the available CPU threads among them
        # to avoid oversubscription.
        if threads is None:
            threads = max(1, min(8, (os.cpu_count() or 8) // max(1, world_size)))
        self.threads = encode_threads(threads)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
        self._byte_lengths = tokenizer.byte_lengths
        self._tokens = np.empty(0, dtype=np.int32)
        self._is_text = np.empty(0, dtype=np.bool_)
        # Documents whose tokens are still (partly) in the buffer, as
        # [corpus state to resume AT this document, token count]. Together with
        # _offset -- how many of the head document's tokens are already spent --
        # this also retains compatibility with version-2 cursor checkpoints.
        self._docs = collections.deque()
        self._offset = 0
        self._next_state = None
        self._resume_corpus = None
        self._pending_skip = 0
        self.documents_seen = 0
        self.tokens_produced = 0
        self.batches_produced = 0
        self._exhausted = False
        self._documents = None

    # -- state -------------------------------------------------------------

    def state_dict(self) -> dict:
        """Exact resume including the finite prefetch buffer and counters.

        Version 3 retains the already tokenized document batch. This avoids
        double-counting replayed documents and changing prefetch boundaries
        after recovery; no corpus or token cache is written separately.
        """
        return {
            "format_version": 3,
            "corpus": self._docs[0][0] if self._docs else self._next_state,
            "skip_tokens": self._offset,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "mask_separator": self.mask_separator,
            "documents_seen": self.documents_seen,
            "tokens_produced": self.tokens_produced,
            "batches_produced": self.batches_produced,
            "loop": self.loop,
            "drop_last": self.drop_last,
            "doc_batch": self.doc_batch,
            "next_state": copy.deepcopy(self._next_state),
            "tokens": self._tokens.copy(),
            "is_text": self._is_text.copy(),
            "docs": copy.deepcopy(list(self._docs)),
            "exhausted": self._exhausted,
            "reader_counters": {
                k: getattr(self.reader, k) for k in ("documents_yielded", "lines_blank", "bad_utf8")
            },
        }

    def load_state_dict(self, state):
        version = state.get("format_version")
        if version not in (2, 3):
            raise DataLoaderError(
                f"resume state format_version {state.get('format_version')!r} is not 2 or 3"
            )
        keys = ("sequence_length", "mask_separator")
        if version == 3:
            keys += ("batch_size", "loop", "drop_last", "doc_batch")
        for key in keys:
            if state.get(key) != getattr(self, key):
                raise DataLoaderError(
                    f"resume state has {key}={state.get(key)!r} but this loader "
                    f"has {getattr(self, key)!r}; the packing would not line up"
                )
        self._resume_corpus = state["corpus"]
        self._pending_skip = int(state["skip_tokens"])
        self._tokens = np.empty(0, dtype=np.int32)
        self._is_text = np.empty(0, dtype=np.bool_)
        self._docs.clear()
        self._offset = 0
        self.documents_seen = state.get("documents_seen", 0)
        self.tokens_produced = state.get("tokens_produced", 0)
        self.batches_produced = state.get("batches_produced", 0)
        self._documents = None
        self._next_state = copy.deepcopy(self._resume_corpus)
        self._exhausted = False
        if version == 3:
            self._tokens = np.asarray(state["tokens"], dtype=np.int32).copy()
            self._is_text = np.asarray(state["is_text"], dtype=np.bool_).copy()
            if self._tokens.ndim != 1 or self._tokens.shape != self._is_text.shape:
                raise DataLoaderError("invalid checkpoint token/mask buffer")
            self._docs = collections.deque(copy.deepcopy(state["docs"]))
            self._offset = int(state["skip_tokens"])
            self._pending_skip = 0
            self._next_state = copy.deepcopy(state["next_state"])
            self._resume_corpus = copy.deepcopy(self._next_state)
            self._exhausted = state["exhausted"]
            for key, value in state["reader_counters"].items():
                setattr(self.reader, key, value)
            if self._next_state is not None:
                cursor = self._next_state
                self.reader._position = (cursor["epoch"], cursor["shard"], cursor["line"])

    # -- stream assembly ---------------------------------------------------

    def _append(self, encoded):
        """Append [*ids, eos] per document; returns each document's token count."""
        counts = [len(ids) + 1 for ids in encoded]
        total = sum(counts)
        tokens = np.empty(total, dtype=np.int32)
        is_text = np.ones(total, dtype=np.bool_)
        offset = 0
        for ids in encoded:
            count = len(ids)
            if count:
                tokens[offset : offset + count] = ids
            tokens[offset + count] = self.eos_id
            is_text[offset + count] = False
            offset += count + 1
        self._tokens = np.concatenate((self._tokens, tokens))
        self._is_text = np.concatenate((self._is_text, is_text))
        return counts

    def _flush(self, texts, states):
        if not texts:
            return
        counts = self._append(self.tokenizer.encode_batch(texts, threads=self.threads))
        for state, count in zip(states, counts):
            self._docs.append([state, count])
        texts.clear()
        states.clear()

    def _fill(self, documents, needed):
        """Read and tokenize until the buffer holds `needed` tokens, or runs dry."""
        texts, states = [], []
        while len(self._tokens) < needed and not self._exhausted:
            try:
                document = next(documents)
            except StopIteration:
                self._exhausted = True
                break
            # The reader's state after yielding document N-1 is the position
            # that resumes AT document N, which is what a row starting inside
            # document N needs to record.
            states.append(self._next_state)
            texts.append(document.text)
            self._next_state = self.reader.state()
            self.documents_seen += 1
            if len(texts) >= self.doc_batch:
                self._flush(texts, states)
        self._flush(texts, states)

    def _drop(self, count):
        """Advance the stream by `count` tokens, retiring consumed documents."""
        self._tokens = self._tokens[count:]
        self._is_text = self._is_text[count:]
        self._offset += count
        while self._docs and self._offset >= self._docs[0][1]:
            self._offset -= self._docs[0][1]
            self._docs.popleft()

    def _row(self, offset):
        """One row from the flat stream, following eval.py:62-66."""
        length = self.sequence_length
        inputs = self._tokens[offset : offset + length].astype(np.int64)
        target = slice(offset + 1, offset + length + 1)
        labels = self._tokens[target].astype(np.int64)
        if self.mask_separator:
            labels = np.where(self._is_text[target], labels, IGNORE_INDEX)
        return inputs, labels

    # -- iteration ---------------------------------------------------------

    def batches(self):
        """Yield Batch objects. Infinite while `loop` is set."""
        if self._documents is None:
            self._documents = self.reader.documents(resume=self._resume_corpus)
        documents = self._documents
        length = self.sequence_length
        row_span = length + 1
        if self._pending_skip:
            self._fill(documents, self._pending_skip + row_span)
            if len(self._tokens) < self._pending_skip:
                raise DataLoaderError(
                    f"resume needs to discard {self._pending_skip} tokens but the "
                    f"corpus only produced {len(self._tokens)} from that position; "
                    "the shard list or tokenizer is not the one that wrote this state"
                )
            self._drop(self._pending_skip)
            self._pending_skip = 0
        while True:
            rows_in, rows_out = [], []
            while len(rows_in) < self.batch_size:
                self._fill(documents, row_span)
                if len(self._tokens) >= row_span:
                    inputs, labels = self._row(0)
                    self._drop(length)
                elif self._exhausted and not self.drop_last and len(self._tokens) > 1:
                    # Tail row: pad like eval.py:59-66 rather than drop data.
                    available = len(self._tokens) - 1
                    inputs = np.full(length, self.pad_id, dtype=np.int64)
                    labels = np.full(length, IGNORE_INDEX, dtype=np.int64)
                    part_in, part_out = self._row(0)
                    inputs[:available] = part_in[:available]
                    labels[:available] = part_out[:available]
                    self._drop(len(self._tokens))
                else:
                    break
                rows_in.append(inputs)
                rows_out.append(labels)
            if not rows_in:
                return
            yield self._make_batch(rows_in, rows_out)
            if self._exhausted and len(self._tokens) <= 1:
                return

    def _make_batch(self, rows_in, rows_out):
        inputs = np.stack(rows_in)
        labels = np.stack(rows_out)
        scored = labels != IGNORE_INDEX
        scored_tokens = int(scored.sum())
        scored_bytes = int(self._byte_lengths[labels[scored]].sum())
        padded = int((inputs == self.pad_id).sum()) if not self.drop_last else 0
        self.tokens_produced += int(inputs.size)
        self.batches_produced += 1
        return Batch(
            inputs=inputs,
            labels=labels,
            scored_tokens=scored_tokens,
            scored_bytes=scored_bytes,
            padded_positions=padded,
            state=self.state_dict(),
        )
