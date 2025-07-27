from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer
from tokenizers.decoders import Decoder

from decasing.byte_handlers import text_to_token_str, token_to_bytes


class TokenizerType(str, Enum):
    """Enum representing different types of tokenizers."""

    BPE = "BPE"
    UNIGRAM = "Unigram"
    WORDPIECE = "WordPiece"


class TokenizerModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: TokenizerUnion

    @classmethod
    def from_tokenizer(cls: type[T], tokenizer: Tokenizer) -> T:
        """Create a TokenizerModel from a Tokenizer."""
        tokenizer_json = json.loads(tokenizer.to_str())
        return cls.model_validate(tokenizer_json)


class BaseTokenizerModel(BaseModel):
    """Basic for all tokenizer models."""

    model_config = ConfigDict(extra="allow")


class WordPieceModel(BaseTokenizerModel):
    """Data model representing a WordPiece vocabulary."""

    type: Literal[TokenizerType.WORDPIECE] = TokenizerType.WORDPIECE
    vocab: dict[str, int]

    def lowercase(self, special_tokens: list[str]) -> None:
        """Return a new WordPieceModel with all tokens lowercased."""
        vocab_sequence, _ = zip(*sorted(self.vocab.items(), key=lambda item: item[1]))
        decased_vocab = _decase_vocab(list(vocab_sequence), special_tokens)
        self.vocab = {k: idx for idx, k in enumerate(decased_vocab)}


class BPEModel(BaseTokenizerModel):
    """Data model representing a BPE vocabulary."""

    type: Literal[TokenizerType.BPE] = TokenizerType.BPE
    merges: list[tuple[str, str]]
    vocab: dict[str, int]

    def lowercase(self, special_tokens: list[str]) -> None:
        """Return a new BPEModel with all tokens lowercased."""
        vocab_sequence, _ = zip(*sorted(self.vocab.items(), key=lambda item: item[1]))
        decased_vocab = _decase_vocab(list(vocab_sequence), special_tokens)

        vocab_set = set(decased_vocab)
        merged_tokens = set()
        new_merges = []
        for prefix, suffix in self.merges:
            new_prefix = decased_vocab[self.vocab[prefix]]
            new_suffix = decased_vocab[self.vocab[suffix]]
            if new_prefix + new_suffix not in decased_vocab:
                continue
            new_merges.append((new_prefix, new_suffix))
            merged_tokens.add(new_prefix + new_suffix)
        unmerged_tokens = set(decased_vocab) - merged_tokens
        for token in unmerged_tokens:
            for needle in range(1, len(token) - 1):
                prefix, suffix = token[:needle], token[needle:]
                if prefix in vocab_set and suffix in vocab_set:
                    new_merges.append((prefix, suffix))
        # Sort merges by length of the prefix, longest first
        self.merges = new_merges
        self.vocab = {k: idx for idx, k in enumerate(decased_vocab)}


class UnigramModel(BaseTokenizerModel):
    """Data model representing a Unigram vocabulary."""

    type: Literal[TokenizerType.UNIGRAM] = TokenizerType.UNIGRAM
    vocab: list[tuple[str, float]]

    def lowercase(self, special_tokens: list[str]) -> None:
        """Return a new UnigramModel with all tokens lowercased."""
        decased_vocab = _decase_vocab([token for token, _ in self.vocab], special_tokens)
        probabilities = [prob for _, prob in self.vocab]
        self.vocab = [(token, prob) for token, prob in zip(decased_vocab, probabilities)]


TokenizerUnion = Annotated[BPEModel | UnigramModel | WordPieceModel, Field(discriminator="type")]


T = TypeVar("T", bound=TokenizerModel)


def _determine_collision(token: str, is_byte: bool, vocab: set[str], special_tokens: list[str], seen: set[str]) -> str:
    """Determine the collision token for a given token."""
    if token in special_tokens:
        return token
    if is_byte:
        # Convert token from bytes to a string.
        token_bytes = token_to_bytes(token)
        # If this token isn't a valid standalone UTF-8 sequence, we can't decase it safely.
        try:
            new_token = token_bytes.decode("utf-8")  # strict: raises on partial codepoints
        except UnicodeDecodeError:
            return token
    else:
        new_token = token

    lowered_token = new_token.casefold()

    if is_byte:
        lowered_token = text_to_token_str(lowered_token)

    # If the token changed but the lowered version was already in vocab, we have a collision.
    if (lowered_token != token) and (lowered_token in vocab or lowered_token in seen):
        return token
    # If the token was decoded, we re-encode it to ensure it matches the tokenizer's encoding.
    return lowered_token


def _decase_vocab(vocab_sequence: list[str], special_tokens: list[str]) -> list[str]:
    """
    Lowercase the vocabulary of a tokenizer.

    Helper for BPE and WordPiece models to ensure that the vocabulary is lowercased
    """
    uncased_vocab = []
    seen: set[str] = set()
    vocab_set = set(vocab_sequence)
    for token in vocab_sequence:
        lowered = _determine_collision(token, True, vocab_set, special_tokens, seen)
        uncased_vocab.append(lowered)
        seen.add(lowered)

    return uncased_vocab
