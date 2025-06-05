# Kompetencer
Kompetencer: Fine-grained Skill Classification in Danish Job Postings via Distant Supervision and Transfer Learning

If you use the code, data, guidelines, models from Kompetencer, please include the following reference:

```
@InProceedings{zhang-jensen-plank:2022:LREC,
  author    = {Zhang, Mike  and  Jensen, Kristian N{\o}rgaard  and  Plank, Barbara},
  title     = {Kompetencer: Fine-grained Skill Classification in Danish Job Postings via Distant Supervision and Transfer Learning},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {436--447},
  abstract  = {Skill Classification (SC) is the task of classifying job competences from job postings. This work is the first in SC applied to Danish job vacancy data. We release the first Danish job posting dataset: *Kompetencer* (\_en\_: competences), annotated for nested spans of competences. To improve upon coarse-grained annotations, we make use of The European Skills, Competences, Qualifications and Occupations (ESCO; le Vrang et al., (2014)) taxonomy API to obtain fine-grained labels via distant supervision. We study two setups: The zero-shot and few-shot classification setting. We fine-tune English-based models and RemBERT (Chung et al., 2020) and compare them to in-language Danish models. Our results show RemBERT significantly outperforms all other models in both the zero-shot and the few-shot setting.},
  url       = {https://aclanthology.org/2022.lrec-1.46}
}
```

## Cloning this repo

In `data/`, you can find the skill and knowledge snippets with their distantly supervised labels (English train+dev+test)
and gold labels (Danish train+test).

__[Note 15/04/2022]__: Job postings are currently being de-identified according to GDPR regulations

__[Note 30/05/2022]__: Danish job postings annotated for skills and knowledge can be found here: https://drive.google.com/file/d/1LoGmoz1BKfEaBFXvyMhaTh2PPTYxNJR0/view?usp=sharing

*Note:* The data can now also be found in `data/*`

The data is structured in the `conll` format:
```
Token <\t> Skill-tag <\t> Knowledge-tag

e.g.,
Python <\t> O <\t> B-Knowledge
...
```

For plotting, the current weighted macro-F1 scores are *hardcoded* in the scripts, predictions can be found in `predictions/`


# Models

All models used in this paper can be found at: https://huggingface.co/jjzha

## Running the code

### Installing the requirements

To install all the required packages run the following command

```
pip3 install --user requirements.txt
```
To finetune the models, you can run `run.finetune.sh`

To predict on the test set, you can run 
```
python3 predict.py logs/esco.new.$MODEL.$PARAMETERS.$c/*/model.tar.gz data/ESCO/$TEST.tsv predictions/da_test/$MODEL/$c.out --device 0
```

Where you have to replace `$*` for the correct variable. For example:
```
python3 predict.py logs/esco.new.dajobbert.da_classification.1/*/model.tar.gz data/ESCO/da_test.tsv predictions/da_test/dajobbert/1.out --device 0
```
