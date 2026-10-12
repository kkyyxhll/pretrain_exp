#!/usr/bin/env python3
"""Streaming document reader for the official Dolma3 shards.

Reads `.jsonl` and `.jsonl.zst` through one code path with a flat memory
profile. The eight shards are 22.92 GB compressed and 228.23 GB decompressed,
so documents are streamed without materialising another full corpus.

Three properties matter downstream and are deliberately not configurable:

* `text` is handed out as a python `str`, unmodified. `Document.nbytes` is
  `len(text.encode("utf-8"))` -- the same expression as
  eval/official/eval.py:35, which is the bpb denominator. Nothing here
  normalises, strips or re-encodes text, because any of those would move the
  denominator away from the official one.
* Document identity is `(shard file name, line index)` where the line index
  counts every line in the file, blank ones included. That makes it a pure
  file offset, so a resume point means the same thing at every world size.
* Rank sharding is a stride over that line index, and a line this rank does
  not own is never JSON-decoded. Each rank decodes only its assigned lines.
  Decompression is still paid in full by every rank because the shards'
  zstd frames do not declare their content size and are not seekable.

Encoding policy: a JSON `\\uD800`-style escape decodes to a lone surrogate,
which `text.encode("utf-8")` rejects -- it would kill the official evaluator at
eval.py:35 before tokenization. Such a document is skipped and counted, never
silently repaired: re-encoding with `errors="replace"` would change the byte
count and therefore the bpb denominator. Use `on_bad_utf8="raise"` for a
pre-flight scan. A missing or non-string `text` field is always an error, since
that indicates the record format itself changed.
"""

import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 4 MiB streaming reads. The longest document measured in the corpus
# is about 270 KB, so a 4 MiB block holds many complete lines and the residual
# buffer stays small.
CHUNK = 1 << 22

TEXT_KEY = "text"
BAD_UTF8_POLICIES = ("skip", "raise")


class CorpusError(RuntimeError):
    """A corpus path, record format or encoding problem the caller must resolve."""


def require_zstandard():
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CorpusError(
            "the python zstandard module is required to read .jsonl.zst "
            "(the zstd CLI is absent from this environment)"
        ) from exc
    return zstandard


@contextlib.contextmanager
def open_stream(path):
    """Yield a forward-only binary stream of decompressed bytes."""
    path = Path(path)
    name = path.name
    if not path.exists():
        raise CorpusError(f"{path} does not exist")
    if name.endswith(".jsonl.zst"):
        zstandard = require_zstandard()
        with path.open("rb") as handle:
            # read_across_frames: the official shards are written as multiple
            # zstd frames. A reader that stops at the first frame truncates the
            # corpus silently, which is far worse than failing.
            yield zstandard.ZstdDecompressor().stream_reader(handle, read_across_frames=True)
    elif name.endswith(".jsonl"):
        with path.open("rb") as handle:
            yield handle
    else:
        raise CorpusError(f"{name}: expected a .jsonl or .jsonl.zst file")


def iter_raw_lines(path, *, chunk_size=CHUNK):
    """Yield complete lines as bytes, with no decoding and no parsing.

    The trailing newline is stripped. A final line without a newline is still
    yielded, so a truncated file loses no record that is actually present.
    """
    with open_stream(path) as stream:
        buffer = b""
        while block := stream.read(chunk_size):
            buffer += block
            *complete, buffer = buffer.split(b"\n")
            yield from complete
        if buffer:
            yield buffer


@dataclasses.dataclass(frozen=True)
class Document:
    """One record. `index` counts every line in `shard`, blank lines included."""

    shard: str
    index: int
    epoch: int
    text: str

    @property
    def nbytes(self) -> int:
        """UTF-8 byte count: eval/official/eval.py:35's expression verbatim."""
        return len(self.text.encode("utf-8"))


def _has_surrogate_escape(line: bytes) -> bool:
    """Cheap pre-filter for lines that could decode to a lone surrogate.

    A surrogate can only reach a python str through a JSON `\\uXXXX` escape --
    json.loads rejects malformed raw UTF-8 outright -- and JSON spells that
    escape with a lowercase `u`. So two substring scans cover every case and
    the expensive validation runs on almost no lines.
    """
    return b"\\ud" in line or b"\\uD" in line


class CorpusReader:
    """Streams documents from an ordered list of shards, one rank's share.

    Resume is exact but not cheap: zstd frames here are not seekable, so
    restarting inside a shard re-reads and discards that shard's earlier bytes.
    Discarding skips JSON decoding, so it runs at decompression speed.
    """

    def __init__(
        self,
        paths,
        *,
        rank=0,
        world_size=1,
        text_key=TEXT_KEY,
        on_bad_utf8="skip",
        on_blank="skip",
        loop=False,
        chunk_size=CHUNK,
    ):
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise CorpusError("rank must be an int")
        if not isinstance(world_size, int) or isinstance(world_size, bool):
            raise CorpusError("world_size must be an int")
        if world_size < 1:
            raise CorpusError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise CorpusError(f"rank {rank} is outside [0, {world_size})")
        if on_bad_utf8 not in BAD_UTF8_POLICIES:
            raise CorpusError(
                f"on_bad_utf8 must be one of {BAD_UTF8_POLICIES}, got {on_bad_utf8!r}"
            )
        if on_blank not in ("skip", "raise"):
            raise CorpusError("on_blank must be 'skip' or 'raise'")
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise CorpusError("no shard paths given")
        names = [p.name for p in self.paths]
        if len(set(names)) != len(names):
            # Resume state records a shard by name, so names must be unique.
            raise CorpusError(f"shard file names must be unique, got {names}")
        self.rank = rank
        self.world_size = world_size
        self.text_key = text_key
        self.on_bad_utf8 = on_bad_utf8
        self.on_blank = on_blank
        self.loop = loop
        self.chunk_size = chunk_size
        self.documents_yielded = 0
        self.lines_blank = 0
        self.bad_utf8 = 0
        self._position = None

    def state(self) -> dict | None:
        """Resume point covering every document yielded so far, or None."""
        if self._position is None:
            return None
        epoch, shard, index = self._position
        return {
            "epoch": epoch,
            "shard": shard,
            "line": index,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def _plan(self, resume):
        """Translate a resume dict into (shard offset, line to skip past, epoch)."""
        if resume is None:
            return 0, -1, 0
        for key in ("epoch", "shard", "line", "rank", "world_size"):
            if key not in resume:
                raise CorpusError(f"resume state is missing {key!r}")
        if (resume["rank"], resume["world_size"]) != (self.rank, self.world_size):
            # Replaying another rank's stride would both duplicate and drop
            # documents, so this has to be refused rather than reinterpreted.
            raise CorpusError(
                f"resume state is for rank {resume['rank']}/{resume['world_size']} "
                f"but this reader is rank {self.rank}/{self.world_size}; "
                "a checkpoint can only resume at the world size that wrote it"
            )
        names = [p.name for p in self.paths]
        if resume["shard"] not in names:
            raise CorpusError(
                f"resume state names shard {resume['shard']!r}, which is not in "
                f"this reader's shard list {names}"
            )
        return names.index(resume["shard"]), int(resume["line"]), int(resume["epoch"])

    def documents(self, resume=None):
        """Yield this rank's documents in shard order."""
        shard_offset, skip_through, epoch = self._plan(resume)
        if resume is not None:
            self._position = (epoch, resume["shard"], skip_through)
        while True:
            pass_start = self.documents_yielded
            full_pass = shard_offset == 0 and skip_through == -1
            for shard_index in range(shard_offset, len(self.paths)):
                path = self.paths[shard_index]
                name = path.name
                for line_index, line in enumerate(iter_raw_lines(path, chunk_size=self.chunk_size)):
                    if line_index <= skip_through:
                        continue
                    if line_index % self.world_size != self.rank:
                        continue
                    if not line or line.isspace():
                        if self.on_blank == "raise":
                            raise CorpusError(f"{name} line {line_index}: blank JSONL record")
                        self.lines_blank += 1
                        continue
                    text = self._text_of(line, name, line_index)
                    if text is None:
                        continue
                    self._position = (epoch, name, line_index)
                    self.documents_yielded += 1
                    yield Document(shard=name, index=line_index, epoch=epoch, text=text)
                # Only the shard named by the resume state is entered partway.
                skip_through = -1
            shard_offset = 0
            if not self.loop:
                return
            if full_pass and self.documents_yielded == pass_start:
                raise CorpusError("cannot loop over a corpus containing no documents")
            epoch += 1

    def _text_of(self, line: bytes, shard: str, index: int):
        """Decode one line to text, or None when policy says to skip it."""
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorpusError(f"{shard} line {index}: not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise CorpusError(
                f"{shard} line {index}: record is {type(record).__name__}, not an object"
            )
        try:
            text = record[self.text_key]
        except KeyError:
            raise CorpusError(
                f"{shard} line {index}: record has no {self.text_key!r} field "
                f"(keys: {sorted(record)[:8]})"
            ) from None
        if not isinstance(text, str):
            raise CorpusError(
                f"{shard} line {index}: {self.text_key!r} is {type(text).__name__}, not a string"
            )
        if _has_surrogate_escape(line):
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                if self.on_bad_utf8 == "raise":
                    raise CorpusError(
                        f"{shard} line {index}: {self.text_key!r} contains a lone "
                        f"surrogate that UTF-8 cannot encode ({exc.reason}); this "
                        "document would also abort eval/official/eval.py:35"
                    ) from exc
                self.bad_utf8 += 1
                return None
        return text


def resolve_shards(shards=None, splits=("train",)):
    """Find official shards directly on disk; prefer existing plain JSONL."""
    result = []
    for split in splits or ("train",):
        if split == "train":
            candidates = sorted((REPO_ROOT / "data" / "train").glob("dolma3_mix_50B-*.jsonl*"))
        elif split == "eval_a":
            candidates = sorted((REPO_ROOT / "eval").glob("dolma3_mix_100M_a.jsonl*"))
        else:
            raise CorpusError(f"unknown split: {split}")
        by_name = {}
        for path in candidates:
            if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.zst")):
                continue
            name = path.name.removesuffix(".zst")
            if name not in by_name or path.suffix == ".jsonl":
                by_name[name] = path
        result.extend(by_name.values())
    if shards:
        selected = []
        for shard in shards:
            matches = [p for p in result if shard in p.name]
            if not matches:
                raise CorpusError(f"shard not found: {shard}")
            selected.extend(matches)
        result = selected
    if not result:
        raise CorpusError("no data files found; download the shards as described in README.md")
    return sorted(set(result))


def corpus_texts(reader, *, max_bytes, report_every=None, stream=sys.stderr):
    """Yield texts until `max_bytes` of UTF-8 source has been handed over."""
    served = 0
    documents = 0
    started = time.monotonic()
    for document in reader.documents():
        text = document.text
        served += document.nbytes
        documents += 1
        yield text
        if report_every and documents % report_every == 0:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"  fed {documents} docs / {served / 1e6:.0f} MB "
                f"({served / 1e6 / elapsed:.0f} MB/s)",
                file=stream,
                flush=True,
            )
        if served >= max_bytes:
            break
    reader.fed_bytes = served
    reader.fed_documents = documents


def guard_eval_split(paths, allow_eval_set):
    """Refuse to fit a tokenizer on the A set unless told to, loudly.

    Repo policy (AGENTS.md): the A set is for validation and model selection.
    The rule text grants that use; it does not spell out a prohibition on
    tokenizer fitting, so this is the repo's reading of the grant and not a
    quoted official rule. The guard exists because the consequence -- a
    submitted tokenizer fitted on evaluation data -- is not recoverable after
    the deadline.
    """
    suspects = [
        p.name for p in paths if "_a." in p.name or "_b." in p.name or "eval" in p.parts[-2:]
    ]
    if suspects and not allow_eval_set:
        raise CorpusError(
            f"refusing to train on {suspects}: these look like evaluation data, "
            "and repo policy restricts the A set to validation and model "
            "selection. Pass --allow-eval-set to override, and record it in the "
            "model card if you do."
        )
    return suspects


def sample_documents(count, *, shards=None, splits=("train",), paths=None):
    """Read a sample of real documents for verification."""
    resolved = [Path(p) for p in paths] if paths else resolve_shards(shards, splits)
    reader = CorpusReader(resolved, on_bad_utf8="skip")
    texts = []
    for document in reader.documents():
        texts.append(document.text)
        if len(texts) >= count:
            break
    if not texts:
        raise CorpusError(f"could not read any document from {resolved}")
    return texts


def scan_special_collisions(paths, names, *, max_bytes=0):
    """Byte-scan corpus files for literal occurrences of special-token strings.

    A document that literally contains "<|bos|>" tokenizes as one special id
    under HuggingFace but as ordinary text under tiktoken, so train-time and
    eval-time tokenization diverge there. Callers pass the special names (e.g.
    `tokenizer.DEFAULT_SPECIALS`); the B set is never available to scan.
    """
    needles = {name: name.encode("utf-8") for name in names}
    counts = {name: 0 for name in names}
    # Carry over the longest needle minus one byte so an occurrence straddling
    # a block boundary is seen.
    keep = max((len(n) for n in needles.values()), default=1) - 1
    scanned = 0
    truncated = False
    for path in paths:
        with open_stream(path) as stream:
            tail = b""
            while block := stream.read(1 << 22):
                scanned += len(block)
                window = tail + block
                for name, needle in needles.items():
                    position = window.find(needle, max(0, keep - len(needle) + 1))
                    while position != -1:
                        counts[name] += 1
                        position = window.find(needle, position + 1)
                tail = window[-keep:] if keep else b""
                if max_bytes and scanned >= max_bytes:
                    truncated = True
                    break
        if truncated:
            break
    return {
        "scanned_decompressed_bytes": scanned,
        "complete_scan": not (max_bytes and scanned >= max_bytes),
        "shards": [p.name for p in paths],
        "occurrences": counts,
        "clean": sorted(n for n, c in counts.items() if c == 0),
        "colliding": {n: c for n, c in counts.items() if c},
    }
