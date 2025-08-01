import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator

from decasing import decase_tokenizer

if __name__ == "__main__":
    model_name = "nomic-ai/modernbert-embed-base"
    suffix = model_name.split("/")[-1]
    dir_name = Path("results") / suffix
    dir_name.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    """original_model = SentenceTransformer(model_name)
    original_model.tokenizer.model_max_length = 512
    original_model.max_seq_length = 512

    evaluator = NanoBEIREvaluator(
        batch_size=4
    )
    results = evaluator(original_model)
    json.dump(results, open(dir_name / "original.json", "w"), indent=4)"""

    new_model = SentenceTransformer(model_name)
    new_model.tokenizer = decase_tokenizer(new_model.tokenizer, make_greedy=True)
    new_model.tokenizer.model_max_length = 512
    new_model.max_seq_length = 512

    evaluator = NanoBEIREvaluator(batch_size=4)
    results_decase = evaluator(new_model)
    json.dump(results_decase, open(dir_name / "decase_greedy.json", "w"), indent=4)

    """lower_model = SentenceTransformer(model_name)
    lower_model[0].do_lower_case = True
    lower_model.tokenizer.model_max_length = 512
    lower_model.max_seq_length = 512

    evaluator = NanoBEIREvaluator(
        batch_size=4
    )
    results_lower = evaluator(lower_model)
    json.dump(results_lower, open(dir_name / "lower.json", "w"), indent=4)"""
