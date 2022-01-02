from typing import Dict, List
import copy

from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from machamp.dataset_readers.lemma_edit import gen_lemma_rule
from machamp.dataset_readers.reader_utils import _clean_text, lines2data


def read_classification(dataset, config, path, is_train, max_sents, token_indexers, tokenizer):
    """
    Reads classification data, meaning that it reads input text from N columns, and
    a corresponding label from a specific column.
    """
    data = []
    sent_idxs = config['sent_idxs']
    for sent_counter, data_instance in enumerate(lines2data(path, config['skip_first_line'])):
        if max_sents != 0 and sent_counter > max_sents:
            break

        # We use the following format 
        # input: <CLS> sent1 <SEP> sent2 <SEP> sent3 ...
        # type_ids: 0 0 .. 1 1 .. 0 0 .. 1 1 ..
        start_token = tokenizer.tokenize(tokenizer.tokenizer.cls_token)[0]
        end_token = tokenizer.tokenize(tokenizer.tokenizer.sep_token)[0]
        setattr(end_token, 'type_id', 0)
        setattr(start_token, 'type_id', 0)
        full_input = [start_token]

        for counter, sent_idx in enumerate(sent_idxs):
            new_sent = tokenizer.tokenize(data_instance[sent_idx].strip()) + [copy.deepcopy(end_token)] 
            for word in new_sent:
                setattr(word, 'type_id', counter%2)

            full_input.extend(new_sent)

        if len(full_input) == 2:
            full_input.insert(1, tokenizer.tokenize(tokenizer.tokenizer.unk_token))

        for token in full_input:
            setattr(token, 'ent_type_', 'TOKENIZED')

        input_field = TextField(full_input, token_indexers)
        instance = Instance({'tokens': input_field})
        dataset_embeds = []
        if 'dec_dataset_embed_idx' in config and config['dec_dataset_embed_idx'] != -1:
            instance.add_field('dec_dataset_embeds', SequenceLabelField([data_instance[config['dec_dataset_embed_idx']] for _ in full_input]), input_field, label_namespace= 'dec_dataset_embeds')
        if 'enc_dataset_embed_idx' in config and config['enc_dataset_embed_idx'] != -1:
            instance.add_field('enc_dataset_embeds', SequenceLabelField([data_instance[config['enc_dataset_embed_idx']] for _ in full_input]), input_field, label_namespace= 'dec_dataset_embeds')

        col_idxs = {}
        for task in config['tasks']:
            task_idx = config['tasks'][task]['column_idx']
            task_type = config['tasks'][task]['task_type']
            col_idxs[task] = task_idx
            if task_type == 'classification':
                instance.add_field(task, LabelField(data_instance[task_idx], label_namespace=task))
            else:
                logger.error('Task type ' + task_type + ' for task ' + task + ' in dataset ' +
                             dataset + ' is unknown')

        metadata = {}
        # the other tokens field will often only be available as word-ids, so we save a copy
        metadata['tokens'] = full_input
        metadata["full_data"] = data_instance
        metadata["col_idxs"] = col_idxs
        metadata['is_train'] = is_train
        metadata['no_dev'] = False
        instance.add_field('metadata', MetadataField(metadata))

        data.append(instance)
    return data


