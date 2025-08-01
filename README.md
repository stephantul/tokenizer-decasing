# decasing

This is a small module to _decase_ a tokenizer. A decased tokenizer is a tokenizer that lowercases all incoming text, but which also does not have any lowercase tokens; decasing removes all case from the model, internall as well as externally. Decasing a tokenizer works with any model or tokenizer.

The example below demonstrates how to use it:

```python
from transformers import AutoTokenizer

from decasing import decase_tokenizer

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
decased = decase_tokenizer(tokenizer)

tokens = ["amsterdam", "Amsterdam"]

print([tokenizer.tokenize(token, add_special_tokens=False) for token in tokens])
# ['▁am', 'ster', 'dam'], ['▁Amsterdam']

print([decased.tokenize(token, add_special_tokens=False) for token in tokens])
# [['▁amsterdam'], ['▁amsterdam']]
```

If we had just lower-cased the token before sending it to the original tokenizer, "amsterdam" would have been tokenized as `['▁am', 'ster', 'dam']`. With a decased tokenizer, the original uppercase token is still used by the model.

# Installation

Not on pip yet:

```bash
pip install git+https://github.com/stephantul/decasing.git
```

## Motivation

Here's a couple of facts:

1. Users don't tend to care much about case. Many users will type "apple" and expect to find the company "Apple". We model case consistently, but case is _used_ inconsistently.
2. Setting aside affect (e.g., using upper-case to denote anger), casing is a marginal semantic phenomenon. There are very few cases in which cased tokens really change the meaning of a sentence.
3. Cased tokens take up a lot of space. The ModernBERT tokenizer, for example consists of 25% cased tokens. This leads to weird segmentations: e.g., `"humph" -> ["hum", "ph]` but `"Humph" -> ["Humph"]`. Why is `Humph` a token?
4. Decased tokenizers generate fewer tokens per sequence

# Author

Stéphan Tulkens

# License

MIT
