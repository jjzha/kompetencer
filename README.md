# Kompetencer
Kompetencer: Fine-grained Skill Classification in Danish Job Postings via Distant Supervision and Transfer Learning

**README is in progress.**

## Cloning this repo

In `data/`, you can find the skill and knowledge snippets with their distantly supervised labels (English train+dev+test)
and gold labels (Danish train+test).

**We release the full job postings with their annotations upon acceptance.**

[Note 15/04/2022]: Job postings are currently being de-identified according to GDPR regulations

The current weighted macro-F1 scores are *hardcoded* in the scripts, predictions can be found in `predictions/`

**We will release our DaJobBERT model upon acceptance.**

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
