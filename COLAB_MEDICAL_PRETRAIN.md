# Text corpus ingestion

The text pretraining source for this project is [Common Pile PubMed](https://huggingface.co/datasets/common-pile/pubmed). This branch does not filter documents by chest relevance; chest focus applies to the separate vision dataset.

See [COLAB_MEDICAL_PRETRAIN_FULL.md](COLAB_MEDICAL_PRETRAIN_FULL.md) for the test and full-ingestion commands. The ingestion script is `data/prepare_common_pile_pubmed.py`; the older MedRAG/scientific-papers download scripts were removed from this branch.

Do not commit corpus shards, model checkpoints, or tokenizer outputs to Git. Store those artifacts on Google Drive.
