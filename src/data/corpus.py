"""Read compressed documents without changing text or UTF-8 byte accounting."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, TypedDict

import zstandard

READ_CHUNK_BYTES = 1 << 22
TEXT_FIELD = "text"
RecordPolicy = Literal["skip", "raise"]


class CorpusState(TypedDict):
    """Identify the last yielded document and its rank ownership."""

    epoch: int
    shard: str
    line: int
    rank: int
    world_size: int


class CorpusError(RuntimeError):
    """Report invalid corpus paths, records, or encoding."""


@contextlib.contextmanager
def open_stream(shard_path: str | Path) -> Iterator[BinaryIO | zstandard.ZstdDecompressionReader]:
    """Open a forward-only decompressed byte stream."""
    shard_path = Path(shard_path)
    shard_name = shard_path.name
    if not shard_path.exists():
        raise CorpusError(f"{shard_path} does not exist")
    elif shard_name.endswith(".jsonl.zst"):
        with shard_path.open("rb") as compressed_stream:
            # note (kkyyxhll): Official shards contain multiple frames; stopping early loses records.
            yield zstandard.ZstdDecompressor().stream_reader(
                compressed_stream, read_across_frames=True
            )
    elif shard_name.endswith(".jsonl"):
        with shard_path.open("rb") as document_stream:
            yield document_stream
    else:
        raise CorpusError(f"{shard_name}: expected a .jsonl or .jsonl.zst file")


def iter_raw_lines(
    shard_path: str | Path, *, chunk_size: int = READ_CHUNK_BYTES
) -> Iterator[bytes]:
    """Yield newline-stripped records, including an unterminated final record."""
    with open_stream(shard_path) as document_stream:
        pending_bytes = b""
        while chunk_bytes := document_stream.read(chunk_size):
            pending_bytes += chunk_bytes
            *complete_lines, pending_bytes = pending_bytes.split(b"\n")
            yield from complete_lines
        if pending_bytes:
            yield pending_bytes
        else:
            pass


@dataclass(frozen=True, kw_only=True)
class Document:
    """Store original text and its stable shard, line, and corpus-pass identity."""

    shard: str
    index: int
    epoch: int
    text: str

    @property
    def nbytes(self) -> int:
        """Count original UTF-8 bytes for the established document interface."""
        return len(self.text.encode("utf-8"))


class CorpusReader:
    """Yield one rank's documents and resume by replaying compressed bytes."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        rank: int = 0,
        world_size: int = 1,
        text_key: str = TEXT_FIELD,
        on_bad_utf8: RecordPolicy = "skip",
        on_blank: RecordPolicy = "skip",
        loop: bool = False,
        chunk_size: int = READ_CHUNK_BYTES,
    ) -> None:
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise CorpusError("rank must be an int")
        elif not isinstance(world_size, int) or isinstance(world_size, bool):
            raise CorpusError("world_size must be an int")
        elif world_size < 1:
            raise CorpusError(f"world_size must be >= 1, got {world_size}")
        elif not 0 <= rank < world_size:
            raise CorpusError(f"rank {rank} is outside [0, {world_size})")
        elif on_bad_utf8 not in ("skip", "raise"):
            raise CorpusError(f"on_bad_utf8 must be 'skip' or 'raise', got {on_bad_utf8!r}")
        elif on_blank not in ("skip", "raise"):
            raise CorpusError("on_blank must be 'skip' or 'raise'")
        else:
            pass
        self.paths: list[Path] = [Path(shard_path) for shard_path in paths]
        shard_names = [shard_path.name for shard_path in self.paths]
        if not self.paths:
            raise CorpusError("no shard paths given")
        elif len(set(shard_names)) != len(shard_names):
            raise CorpusError(f"shard file names must be unique, got {shard_names}")
        else:
            pass
        self.rank: int = rank
        self.world_size: int = world_size
        self.text_key: str = text_key
        self.on_bad_utf8: RecordPolicy = on_bad_utf8
        self.on_blank: RecordPolicy = on_blank
        self.loop: bool = loop
        self.chunk_size: int = chunk_size
        self.documents_yielded: int = 0
        self.lines_blank: int = 0
        self.bad_utf8: int = 0
        self.position: tuple[int, str, int] | None = None

    def state(self) -> CorpusState | None:
        """Return the cursor covering every document yielded so far."""
        if self.position is None:
            return None
        else:
            corpus_epoch, shard_name, line_index = self.position
            return {
                "epoch": corpus_epoch,
                "shard": shard_name,
                "line": line_index,
                "rank": self.rank,
                "world_size": self.world_size,
            }

    def resume_plan(self, resume: CorpusState | None) -> tuple[int, int, int]:
        """Resolve a checkpoint to shard offset, skipped line, and corpus pass."""
        if resume is None:
            return 0, -1, 0
        else:
            for state_field in ("epoch", "shard", "line", "rank", "world_size"):
                if state_field not in resume:
                    raise CorpusError(f"resume state is missing {state_field!r}")
                else:
                    pass
            shard_names = [shard_path.name for shard_path in self.paths]
            if (resume["rank"], resume["world_size"]) != (self.rank, self.world_size):
                raise CorpusError(
                    f"resume state is for rank {resume['rank']}/{resume['world_size']} "
                    f"but this reader is rank {self.rank}/{self.world_size}; "
                    "a checkpoint can only resume at the world size that wrote it"
                )
            elif resume["shard"] not in shard_names:
                raise CorpusError(
                    f"resume state names shard {resume['shard']!r}, which is not in "
                    f"this reader's shard list {shard_names}"
                )
            else:
                return (
                    shard_names.index(resume["shard"]),
                    int(resume["line"]),
                    int(resume["epoch"]),
                )

    def documents(self, resume: CorpusState | None = None) -> Iterator[Document]:
        """Yield the rank's original documents in shard order."""
        shard_offset, skipped_line_index, corpus_epoch = self.resume_plan(resume)
        if resume is not None:
            self.position = (corpus_epoch, resume["shard"], skipped_line_index)
        else:
            pass
        while True:
            starting_document_count = self.documents_yielded
            is_complete_pass = shard_offset == 0 and skipped_line_index == -1
            for shard_index in range(shard_offset, len(self.paths)):
                shard_path = self.paths[shard_index]
                shard_name = shard_path.name
                for line_index, line_bytes in enumerate(
                    iter_raw_lines(shard_path, chunk_size=self.chunk_size)
                ):
                    if line_index <= skipped_line_index:
                        continue
                    elif line_index % self.world_size != self.rank:
                        continue
                    elif not line_bytes or line_bytes.isspace():
                        if self.on_blank == "raise":
                            raise CorpusError(f"{shard_name} line {line_index}: blank JSONL record")
                        else:
                            self.lines_blank += 1
                            continue
                    else:
                        document_text = self.decode_text(line_bytes, shard_name, line_index)
                        if document_text is None:
                            continue
                        else:
                            self.position = (corpus_epoch, shard_name, line_index)
                            self.documents_yielded += 1
                            yield Document(
                                shard=shard_name,
                                index=line_index,
                                epoch=corpus_epoch,
                                text=document_text,
                            )
                skipped_line_index = -1
            shard_offset = 0
            if not self.loop:
                return
            elif is_complete_pass and self.documents_yielded == starting_document_count:
                raise CorpusError("cannot loop over a corpus containing no documents")
            else:
                corpus_epoch += 1

    def decode_text(self, line_bytes: bytes, shard_name: str, line_index: int) -> str | None:
        """Decode a record, rejecting invalid text unless the skip policy is explicit."""
        try:
            decoded_record: object = json.loads(line_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as encoding_error:
            raise CorpusError(
                f"{shard_name} line {line_index}: not valid JSON: {encoding_error}"
            ) from encoding_error
        if not isinstance(decoded_record, dict):
            raise CorpusError(
                f"{shard_name} line {line_index}: record is "
                f"{type(decoded_record).__name__}, not an object"
            )
        elif self.text_key not in decoded_record:
            raise CorpusError(
                f"{shard_name} line {line_index}: record has no {self.text_key!r} field "
                f"(keys: {sorted(decoded_record)[:8]})"
            )
        else:
            document_text: object = decoded_record[self.text_key]
            if not isinstance(document_text, str):
                raise CorpusError(
                    f"{shard_name} line {line_index}: {self.text_key!r} is "
                    f"{type(document_text).__name__}, not a string"
                )
            else:
                # note (kkyyxhll): Only JSON escapes can introduce lone surrogates after decoding.
                if b"\\ud" in line_bytes or b"\\uD" in line_bytes:
                    try:
                        document_text.encode("utf-8")
                    except UnicodeEncodeError as encoding_error:
                        if self.on_bad_utf8 == "raise":
                            raise CorpusError(
                                f"{shard_name} line {line_index}: {self.text_key!r} contains a lone "
                                f"surrogate that UTF-8 cannot encode ({encoding_error.reason}); "
                                "this document would also abort the official evaluator"
                            ) from encoding_error
                        else:
                            self.bad_utf8 += 1
                            return None
                else:
                    pass
                return document_text
