This repository is implementation of an arxiv paper "Cross-lingual Transfer Learning for Fake News Detector in a Low-Resource Language" (link: https://arxiv.org/abs/2208.12482).

This code is a modified verson of ADAN (https://github.com/ccsasuke/adan).

Input data structure is given below:

## FILE STRUCTURE

```bash
data
├── embedding
│   ├── BNE
│   │   ├── koEnBNE.txt
│   │   ├── vectors-en.txt
│   │   └── vectors-ko.txt
│   └── BWE
│       ├── koEnBWE.txt
│       ├── vectors-en.txt
│       └── vectors-ko.txt
└── NAIVE
    ├── Eng
    │   ├── en_dev_data.txt
    │   ├── en_dev_label.txt
    │   ├── en_test_data.txt
    │   ├── en_test_label.txt
    │   ├── en_train_data.txt
    │   ├── en_train_label.txt
    │   ├── readMe
    │   ├── reference
    │   │   ├── org
    │   │   │   ├── en_test_label.txt
    │   │   │   ├── en_train_label.txt
    │   │   │   ├── org_ner_dev.txt
    │   │   │   ├── org_ner_test.txt
    │   │   │   └── org_ner_train.txt
    │   │   ├── person
    │   │   │   ├── en_test_label.txt
    │   │   │   ├── en_train_label.txt
    │   │   │   ├── person_ner_dev.txt
    │   │   │   ├── person_ner_test.txt
    │   │   │   └── person_ner_train.txt
    │   │   ├── personOrg
    │   │   │   ├── en_test_label.txt
    │   │   │   ├── en_train_label.txt
    │   │   │   ├── personOrg_ner_dev.txt
    │   │   │   ├── personOrg_ner_test.txt
    │   │   │   └── personOrg_ner_train.txt
    │   │   └── shuffle
    │   │       ├── en_test_label.txt
    │   │       ├── en_train_label.txt
    │   │       ├── shuffle_person_ner_dev.txt
    │   │       ├── shuffle_person_ner_test.txt
    │   │       └── shuffle_person_ner_train.txt
    │   └── unbalanced
    │       ├── en_train_data.txt
    │       ├── en_train_label.txt
    │       └── readMe
    └── Kor
        ├── ko_data.txt
        ├── ko_label.txt
        ├── readMe
        └── reference
            ├── ko_label.txt
            ├── org_ner.txt
            ├── person_ner.txt
            ├── personOrg_ner.txt
            └── shuffle
                ├── ko_label.txt
                ├── shuffle_org_ner.txt
                ├── shuffle_person_ner.txt
                └── shuffle_personOrg_ner.txt
```
