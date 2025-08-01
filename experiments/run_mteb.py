import logging
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

from decasing import decase_tokenizer

if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-small"
    suffix = model_name.split("/")[-1]
    dir_name = Path("results/mteb") / suffix
    dir_name.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    # Define the sentence-transformers model name
    tasks = mteb.get_tasks(task_types=["Clustering"], languages=["eng"], exclusive_language_filter=True)
    # tasks = [task for task in tasks if task.metadata.name not in {"DigikalamagClustering", "HamshahriClustring"}]
    evaluation = mteb.MTEB(tasks=tasks)

    model = SentenceTransformer(model_name)
    model.tokenizer.model_max_length = 512
    model.max_seq_length = 512
    results = evaluation.run(model, output_folder=str(dir_name))

    model = SentenceTransformer(model_name)
    model.tokenizer.model_max_length = 512
    model.max_seq_length = 512
    model.tokenizer = decase_tokenizer(model.tokenizer)
    results = evaluation.run(model, output_folder=f"{str(dir_name)}_decased")

    model = SentenceTransformer(model_name)
    model[0].do_lower_case = True
    model.tokenizer.model_max_length = 512
    model.max_seq_length = 512

    results = evaluation.run(model, output_folder=f"{str(dir_name)}_lowered")
