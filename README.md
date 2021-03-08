# Automatic Paraphrase Dataset Augmentation
This repository includes data and code for implementing the paper **[Finding Friends and Flipping Frenemies: Automatic Paraphrase Dataset Augmentation Using Graph Theory](https://arxiv.org/abs/2011.01856)**.

## Dependencies
You can install all the required packages by running the following command: \
```python -m pip install -r requirements.txt```

## Datasets
**[Quora Question Pairs](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)** \
We used the train/dev splits from the GLUE benchmark, which you can download from **[here](https://gluebenchmark.com/tasks)**.

## Generating Augmented QQP Dataset
```python generate_qqp_datasets.py -o OUTPUT_DIR -d [original_flipped | augmented | augmented_flipped] ```

## Bibtex
```
@inproceedings{chen-etal-2020-finding,
    title = "Finding {F}riends and Flipping Frenemies: Automatic Paraphrase Dataset Augmentation Using Graph Theory",
    author = "Chen, Hannah  and
      Ji, Yangfeng  and
      Evans, David",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.426",
    doi = "10.18653/v1/2020.findings-emnlp.426",
    pages = "4741--4751"
}
```