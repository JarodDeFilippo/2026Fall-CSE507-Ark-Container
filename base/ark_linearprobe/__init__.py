"""The lab's embedding linear probe (Ark_Plus/Linearprobing/main_linearprobing.py) as shipped, plus the ported additions that run it on B2's cached embeddings: see NOTICE.
Run the modules from base/ (the container's working directory) as `python -m ark_linearprobe.<module>`: base/ must come first on sys.path, because the vendored files import their
siblings relatively, while accelerator, models and utils (the container's, not the vendored utils.py) are top-level modules of base/."""
