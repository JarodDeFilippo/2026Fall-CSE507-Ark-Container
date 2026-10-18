"""Finetuning-side pieces of B2 (the lab's embedding linear probe on a backbone's cached embeddings): the vendored official dataset classes and transform (dataloader.py), the
extractor that writes the embedding cache (extract_embeddings.py) and the cache format and label rules the probe's dataset modules share with it (embedding_mode.py); see NOTICE.
The fit itself is ark_linearprobe/. Run the modules from base/ (the container's working directory) as `python -m ark_finetune.<module>`: base/ must come first on sys.path, because the
vendored dataloader.py is imported relatively here, while accelerator, models and utils (the container's) are top-level modules of base/."""
