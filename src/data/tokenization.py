"""Count source bytes and choose bounded tokenizer worker threads."""

from __future__ import annotations

import os

from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import bytes_to_unicode

MAX_ENCODING_THREADS = 8
BYTE_VALUE_COUNT = 256


class TokenizerError(RuntimeError):
    """Report invalid worker settings or byte-level vocabulary mappings."""


def encode_threads(requested_thread_count: int | None = None) -> int:
    """Choose the worker count without exceeding the tokenizer thread limit."""
    if requested_thread_count is None:
        return max(1, min(MAX_ENCODING_THREADS, os.cpu_count() or 1))
    elif (
        not isinstance(requested_thread_count, int)
        or isinstance(requested_thread_count, bool)
        or requested_thread_count < 1
    ):
        raise TokenizerError("threads must be a positive integer")
    else:
        return min(requested_thread_count, MAX_ENCODING_THREADS)


def byte_level_alphabet() -> dict[int, str]:
    """Map every byte value to its byte-level vocabulary character."""
    printable_byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    byte_characters = {byte_value: chr(byte_value) for byte_value in printable_byte_values}
    extra_character_count = 0
    for byte_value in range(BYTE_VALUE_COUNT):
        if byte_value not in byte_characters:
            byte_characters[byte_value] = chr(BYTE_VALUE_COUNT + extra_character_count)
            extra_character_count += 1
        else:
            pass
    assert len(set(byte_characters.values())) == BYTE_VALUE_COUNT
    return byte_characters


def check_byte_level_alphabet(byte_characters: dict[int, str]) -> list[str]:
    """Verify byte accounting against both installed tokenizer libraries."""
    if bytes_to_unicode() != byte_characters:
        raise TokenizerError("the local byte-level alphabet disagrees with transformers")
    elif set(ByteLevel.alphabet()) != set(byte_characters.values()):
        raise TokenizerError("the local byte-level alphabet disagrees with tokenizers")
    else:
        return ["transformers", "tokenizers"]


def hf_token_bytes(tokenizer: PreTrainedTokenizerFast) -> list[bytes]:
    """Invert vocabulary characters without replacing partial UTF-8 token bytes."""
    byte_characters = byte_level_alphabet()
    check_byte_level_alphabet(byte_characters)
    character_byte_values = {
        character: byte_value for byte_value, character in byte_characters.items()
    }
    added_token_pieces = set(tokenizer.get_added_vocab())
    token_pieces = {
        token_id: token_piece for token_piece, token_id in tokenizer.get_vocab().items()
    }
    token_source_bytes = []
    for token_id in range(len(tokenizer)):
        if token_id not in token_pieces:
            raise TokenizerError(f"vocabulary has no entry for id {token_id}")
        else:
            token_piece = token_pieces[token_id]
            if token_piece in added_token_pieces:
                token_source_bytes.append(b"")
            else:
                try:
                    token_source_bytes.append(
                        bytes(character_byte_values[character] for character in token_piece)
                    )
                except KeyError as vocabulary_error:
                    raise TokenizerError(
                        f"token id {token_id} ({token_piece!r}) contains character "
                        f"{vocabulary_error} which is not in the byte-level alphabet"
                    ) from None
    return token_source_bytes
