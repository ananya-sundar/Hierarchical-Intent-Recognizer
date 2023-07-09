import torch

from torchtext import data
from transformers import AutoTokenizer

from utils import load_vocab


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def tokenize_and_cut(sentence):
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens


def init_fields():
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id

    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)

    DOMAIN = data.LabelField(dtype=torch.long, batch_first=True)
    #SCOPE = data.LabelField(dtype=torch.long, batch_first=True)
    INTENT = data.LabelField(dtype=torch.long, batch_first=True)

    FIELDS = [('text', TEXT), ('domain', DOMAIN), ('intent', INTENT)]
    #FIELDS = [('text', TEXT), ('scope', SCOPE), ('intent', INTENT)]
    
    return FIELDS, DOMAIN, INTENT
    #return FIELDS, SCOPE, INTENT


def get_iterator(file_path, batch_size, device, train=False):
    FIELDS, DOMAIN, INTENT = init_fields()
    #FIELDS, SCOPE, INTENT = init_fields()

    if train:
        dataset = data.TabularDataset(path=file_path, format='csv', fields=FIELDS, skip_header=True)
        DOMAIN.build_vocab(dataset)
        #SCOPE.build_vocab(dataset)
        INTENT.build_vocab(dataset)

        #No. of unique tokens in label
        print("Size of scope vocabulary:", len(DOMAIN.vocab))
        torch.save(DOMAIN.vocab, 'scope_vocab.pkl')
        
        #print("Size of scope vocabulary:", len(SCOPE.vocab))
        #torch.save(SCOPE.vocab, 'scope_vocab.pkl')
        
        
        print("Size of intent vocabulary:", len(INTENT.vocab))
        torch.save(INTENT.vocab, 'intent_vocab.pkl')

        iterator = data.Iterator(dataset, batch_size, device=device, train=True, shuffle=True, sort=False)
        #iterator = data.Iterator(dataset, batch_size, device=device, repeat=True, train=True, shuffle=True, sort=True,
        #                         sort_key=lambda x: len(x.text), sort_within_batch=True)
        return iterator, len(DOMAIN.vocab), len(INTENT.vocab)
        #return iterator, len(SCOPE.vocab), len(INTENT.vocab)
    
    else:
        #DOMAIN.vocab = load_vocab('domain_vocab.pkl')
        DOMAIN.vocab = load_vocab('scope_vocab.pkl')
        INTENT.vocab = load_vocab('intent_vocab.pkl')
        dataset = data.TabularDataset(path=file_path, format='csv', fields=FIELDS, skip_header=True)
        iterator = data.Iterator(dataset, batch_size, device=device, train=False, shuffle=False, sort=False)
        return iterator


if __name__ == '__main__':
    #train_iter, _, _ = get_iterator('dataset/oos_full/train.csv', 32, 'cuda', train=True)
    #valid_iter = get_iterator('dataset/oos_full/valid.csv', 32, 'cuda', train=False)
    train_iter, _, _ = get_iterator('dataset/train.csv', 32, 'cuda', train=True)
    valid_iter = get_iterator('dataset/valid.csv', 32, 'cuda', train=False)
    for batch in train_iter:
        print(batch.text)
        print(batch.domain)
        #print(batch.scope)
        print(batch.intent)
        break

    for batch in valid_iter:
        print(batch.text)
        print(batch.domain)
        #print(batch.scope)
        print(batch.intent)
        break
