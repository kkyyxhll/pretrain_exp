"""Minimal helpers for the pinned HF tokenizer and compressed stream."""

import os


class TokenizerError(RuntimeError):
    pass


def encode_threads(requested=None):
    if requested is not None:
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
            raise TokenizerError("threads must be a positive integer")
        return min(requested, 8)
    return max(1, min(8, os.cpu_count() or 1))


def byte_level_alphabet() -> dict:
    """GPT-2's byte-to-printable-character map, as a local definition.

    Defined locally so byte accounting does not depend on a
    private transformers path, then cross-checked against every library copy
    that is importable.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapping = {value: chr(value) for value in printable}
    spare = 0
    for value in range(256):
        if value not in mapping:
            mapping[value] = chr(256 + spare)
            spare += 1
    if len(set(mapping.values())) != 256:
        raise TokenizerError("byte-level alphabet is not a bijection")
    return mapping


def check_byte_level_alphabet(mapping):
    """Confirm the local map equals the library's, wherever one is available."""
    checked = []
    try:
        from transformers.convert_slow_tokenizer import bytes_to_unicode
    except ImportError:
        pass
    else:
        if bytes_to_unicode() != mapping:
            raise TokenizerError(
                "the local byte-level alphabet disagrees with "
                "transformers.convert_slow_tokenizer.bytes_to_unicode"
            )
        checked.append("transformers")
    try:
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError:
        pass
    else:
        if set(ByteLevel.alphabet()) != set(mapping.values()):
            raise TokenizerError(
                "the local byte-level alphabet disagrees with "
                "tokenizers.pre_tokenizers.ByteLevel.alphabet()"
            )
        checked.append("tokenizers")
    return checked


def hf_token_bytes(tokenizer):
    """Per-id source bytes of a loaded HuggingFace tokenizer.

    Inverts the byte-level map over the vocabulary strings. Added special
    tokens carry no source bytes. A `decode([i]).encode()` round-trip is wrong
    here for every token holding a bare UTF-8 continuation byte -- it yields a
    3-byte U+FFFD instead -- and reports bpb too low.
    """
    mapping = byte_level_alphabet()
    check_byte_level_alphabet(mapping)
    inverse = {character: value for value, character in mapping.items()}
    added = set(tokenizer.get_added_vocab())
    by_id = {index: piece for piece, index in tokenizer.get_vocab().items()}
    table = []
    for index in range(len(tokenizer)):
        if index not in by_id:
            raise TokenizerError(f"vocabulary has no entry for id {index}")
        piece = by_id[index]
        if piece in added:
            table.append(b"")
            continue
        try:
            table.append(bytes(inverse[character] for character in piece))
        except KeyError as exc:
            raise TokenizerError(
                f"token id {index} ({piece!r}) contains character {exc} which is "
                "not in the byte-level alphabet"
            ) from None
    return table
