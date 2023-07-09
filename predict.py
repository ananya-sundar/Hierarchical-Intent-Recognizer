
import torch

#from argparse import ArgumentParser

from transformers import BertConfig

from dataset import get_iterator
from models import BertClassifier, BertDomainIntent
from utils import load_vocab, get_label, print_metrics, str2bool, set_seed


def run_single(intent_vocab):
    intent_vocab = load_vocab(intent_vocab)
    num_intents = len(intent_vocab)

    config = BertConfig.from_pretrained(pretrained)
    model = BertClassifier(config=config, num_labels=num_intents)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_true_intents, y_pred_intents = [], []

    intent_vocab = load_vocab(intent_vocab)

    if validation:
        data_loader = get_iterator(data_path + '/valid.csv', val_batch_size, device)
    else:
        data_loader = get_iterator(data_path + '/test.csv', test_batch_size, device)

    for batch in data_loader:
        x, y_true_intent = batch.text, batch.intent

        with torch.no_grad():
            intent_out = model(x)

        intent_oos_idx = intent_vocab.stoi['oos'] if 'oos' in intent_vocab.stoi else -1
        y_pred_intent = get_label(intent_out, intent_oos_idx, threshold)

        y_true_intents.extend(y_true_intent.tolist())
        y_pred_intents.extend(y_pred_intent.tolist())

    print_metrics(y_true_intents, y_pred_intents, intent_vocab, prefix='intent: ')


def run_joint(domain_vocab, intent_vocab):
    domain_vocab = load_vocab(domain_vocab)
    #scope_vocab = load_vocab(scope_vocab)
    intent_vocab = load_vocab(intent_vocab)
    num_domains = len(domain_vocab)
    #num_scopes = len(scope_vocab)
    num_intents = len(intent_vocab)

    config = BertConfig.from_pretrained(pretrained)
    model = BertDomainIntent.from_pretrained(pretrained, config=config, num_domains=num_domains, num_intents=num_intents,
                                             subspace=subspace, hierarchy=hierarchy, domain_first=domain_first)

    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    model.eval()

    y_true_domains, y_true_intents, y_pred_domains, y_pred_intents = [], [], [], []
    #y_true_scopes, y_true_intents, y_pred_scopes, y_pred_intents = [], [], [], []
    
    if validation:
        data_loader = get_iterator(data_path + '/valid.csv', val_batch_size, device)
    else:
        data_loader = get_iterator(data_path + '/test.csv', test_batch_size, device)

    for batch in data_loader:
        x, y_true_domain, y_true_intent = batch.text, batch.domain, batch.intent
        #x, y_true_scope, y_true_intent = batch.text, batch.scope, batch.intent
        with torch.no_grad():
            domain_out, intent_out = model(x)
            #scope_out, intent_out = model(x)

        #scope_oos_idx = scope_vocab.stoi['oos'] if 'oos' in scope_vocab.stoi else -1
        domain_oos_idx = domain_vocab.stoi['oos'] if 'oos' in domain_vocab.stoi else -1
        intent_oos_idx = intent_vocab.stoi['oos'] if 'oos' in intent_vocab.stoi else -1
        
        y_pred_domain = get_label(domain_out, domain_oos_idx, threshold)
        #y_pred_scope = get_label(scope_out, scope_oos_idx, threshold)
        y_pred_intent = get_label(intent_out, intent_oos_idx, threshold)

        #y_true_scopes.extend(y_true_scope.tolist())
        #y_pred_scopes.extend(y_pred_scope.tolist())

        y_true_domains.extend(y_true_domain.tolist())
        y_pred_domains.extend(y_pred_domain.tolist())

        y_true_intents.extend(y_true_intent.tolist())
        y_pred_intents.extend(y_pred_intent.tolist())
    
    print("#######################################################################")
    print("***********************Scope Results***********************************")
    print_metrics(y_true_domains, y_pred_domains, domain_vocab, prefix='scope: ')
    #print_metrics(y_true_scopes, y_pred_scopes, scope_vocab, prefix='scope: ')
    print("#######################################################################")
    print("***********************Intent Results**********************************")
    print_metrics(y_true_intents, y_pred_intents, intent_vocab, prefix='intent: ')
    print("#######################################################################")

if __name__ == "__main__":
    #parser = ArgumentParser()
    single= False #"joint model" #"single" #or "joint model"
    validation=False #False #action='store_true', help="validation or testing")
    data_path = "dataset/" # help="dataset path")
    val_batch_size =32 # help="input batch size for validation (default: 32)")
    test_batch_size =32 #help="input batch size for testing (default: 32)")
    model_path= 'checkpoints/best_model_11_intent_acc=0.9875.pt' #"best_model.pt" #help="model path") #'checkpoints/best_model_9_intent_acc=0.9568.pt'
    #domain_vocab="domain_vocab.pkl" # help="path of domain vocabulary")
    scope_vocab = "scope_vocab.pkl"
    intent_vocab="intent_vocab.pkl" #help="path of intent vocabulary")
    pretrained="bert-base-uncased" # help="model or path of pretrained model")
    threshold =0.7 #help="threshold for oos cutoff")
    subspace='false' #help="use subspace layers")
    hierarchy='true' #help="use hierarchical structure")
    domain_first='false' #help="domain representation first")

    #args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    subspace = str2bool(subspace)
    hierarchy = str2bool(hierarchy)
    domain_first = str2bool(domain_first)

    set_seed()

    if single:
        run_single(intent_vocab)
    else:
        run_joint(scope_vocab, intent_vocab)
        #run_joint(domain_vocab, intent_vocab)
