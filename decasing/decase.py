from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer
from tokenizers.normalizers import Lowercase, Sequence
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from decasing.datamodels import TokenizerModel, WordPieceModel


def decase_tokenizer(tokenizer: PreTrainedTokenizerFast, make_greedy: bool = False) -> PreTrainedTokenizerFast:
    """Decase a tokenizer by removing the casing from the backend tokenizer."""
    backend_tokenizer = tokenizer.backend_tokenizer
    special_tokens = tokenizer.all_special_tokens

    new_tokenizer = _decase_vocabulary(backend_tokenizer, special_tokens=special_tokens, make_greedy=make_greedy)
    new_tokenizer = _add_lowercase(new_tokenizer)

    with TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        tokenizer_json_path = str(Path(tmpdir) / "tokenizer.json")
        new_tokenizer.save(tokenizer_json_path)
        vocab_path = Path(tmpdir) / "vocab.txt"
        with open(vocab_path, "w", encoding="utf-8") as f:
            sorted_vocab = sorted(new_tokenizer.get_vocab().items(), key=lambda item: item[1])
            f.writelines(f"{token}\n" for token, _ in sorted_vocab)
        tokenizer = AutoTokenizer.from_pretrained(
            tmpdir,
            use_fast=True,
        )

    return tokenizer


def _add_lowercase(tokenizer: Tokenizer) -> Tokenizer:
    """
    Add a lowercase Normalizer to the tokenizer.

    If the model already has a normalizer, we prepend the Lowercase normalizer to it.
    If it does not have a normalizer, we set the Lowercase normalizer as the only normalizer.

    Note that we don't care if the tokenizer already has a Lowercase normalizer,
    because there are many ways to lowercase. Also lowercasing is not idempotent,
    so we always add a new Lowercase normalizer.

    :param tokenizer: The tokenizer to modify.
    :return: The modified tokenizer with a Lowercase normalizer.
    """
    if tokenizer.normalizer is not None:
        tokenizer.normalizer = Sequence([Lowercase(), tokenizer.normalizer])  # type: ignore
    else:
        tokenizer.normalizer = Lowercase()  # type: ignore
    return tokenizer


def _decase_vocabulary(tokenizer: Tokenizer, special_tokens: list[str], make_greedy: bool) -> Tokenizer:
    """
    Replace the vocabulary in a tokenizer with a decased version.

    This function dumps the tokenizer to a JSON file, modifies the vocabulary to be lowercased,
    and then loads it back into a new Tokenizer object.

    :param tokenizer: The tokenizer to modify.
    :param special_tokens: A list of special tokens that should not be lowercased.
    :param make_greedy: If True, convert the tokenizer to a greedy model by converting it to a WordPiece model.
    :return: A new Tokenizer with the decased vocabulary.
    """
    tokenizer_model = TokenizerModel.from_tokenizer(tokenizer)
    tokenizer_model.lowercase(special_tokens=special_tokens)

    if make_greedy and not isinstance(tokenizer_model.model, WordPieceModel):
        tokenizer_model.model = tokenizer_model.model.make_greedy()

    return Tokenizer.from_str(tokenizer_model.model_dump_json())
