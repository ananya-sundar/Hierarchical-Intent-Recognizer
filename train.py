
import torch
from torch import nn

from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.engine import Engine, Events

from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver, global_step_from_engine

from transformers import AdamW, BertConfig

from dataset import get_iterator
from utils import set_seed, str2bool
from models import BertDomainIntent, MultiTaskLoss
#import matplotlib.pyplot as plt
import pickle

def train():
    set_seed()

    train_loader, num_domains, num_intents = get_iterator(data_path + '/train.csv', train_batch_size, device, train=True)
    val_loader = get_iterator(data_path + '/valid.csv', val_batch_size, device)

    config = BertConfig.from_pretrained(pretrained)

    model = BertDomainIntent.from_pretrained(pretrained, config=config, num_domains=num_domains,
                                             num_intents=num_intents, dropout=dropout,
                                             subspace=subspace, hierarchy=hierarchy,
                                             domain_first=domain_first)

    model.to(device)

    mtl_instance = MultiTaskLoss()
    mtl_instance.to(device)
    params = list(model.parameters()) + list(mtl_instance.parameters())
    optimizer = AdamW(params, lr=lr, correct_bias=True)
    if fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16)

    criterion = nn.CrossEntropyLoss()

    def update(engine, batch):
        model.train()
        mtl_instance.train()

        
        x_text, y_domain, y_intent = batch.text, batch.domain, batch.intent
        y_pred_domain, y_pred_intent = model(x_text)
        
        
        domain_loss = criterion(y_pred_domain, y_domain)
      
        intent_loss = criterion(y_pred_intent, y_intent)

        loss = mtl_instance(domain_loss, intent_loss)
     

        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm)

        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()


    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x_text, y_domain, y_intent = batch.text, batch.domain, batch.intent
            y_pred_domain, y_pred_intent = model(x_text)
            
            
            return {'domain': (y_pred_domain, y_domain), 'intent': (y_pred_intent, y_intent)}
          


    trainer = Engine(update)
    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, lr), (epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    train_evaluator = Engine(inference)
    valid_evaluator = Engine(inference)
    
    train_scope_acc,train_scope_loss, train_intent_acc, train_intent_loss = [], [],[],[]
    val_scope_acc,val_scope_loss, val_intent_acc, val_intent_loss = [], [],[],[]
    
    def attach_metrics(evaluator):
        Accuracy(output_transform=lambda out: out['domain']).attach(evaluator, 'domain_acc')
        
        Accuracy(output_transform=lambda out: out['intent']).attach(evaluator, 'intent_acc')
        Loss(criterion, output_transform=lambda out: out['domain']).attach(evaluator, 'domain_loss')
       
        Loss(criterion, output_transform=lambda out: out['intent']).attach(evaluator, 'intent_loss')

    attach_metrics(train_evaluator)
    attach_metrics(valid_evaluator)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    def log_message(msg_type, epoch, metrics):
        pbar.log_message(
            "{} - Epoch: {}  scope acc/loss: {:.4f}/{:.4f}, intent acc/loss: {:.4f}/{:.4f}".format(
                msg_type, epoch, metrics['domain_acc'], metrics['domain_loss'],
                metrics['intent_acc'], metrics['intent_loss'])
        )
        
        return metrics['domain_acc'], metrics['domain_loss'],metrics['intent_acc'], metrics['intent_loss']
            

    #@trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        scope_acc, scope_loss, intent_acc, intent_loss = log_message('Training', engine.state.epoch, metrics)
        train_scope_acc.append(scope_acc)
        train_scope_loss.append(scope_loss) 
        train_intent_acc.append(intent_acc) 
        train_intent_loss.append(intent_loss)

    #@trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        valid_evaluator.run(val_loader)
        metrics = valid_evaluator.state.metrics
        scope_acc, scope_loss, intent_acc, intent_loss = log_message('Validation', engine.state.epoch, metrics)
        val_scope_acc.append(scope_acc)
        val_scope_loss.append(scope_loss) 
        val_intent_acc.append(intent_acc) 
        val_intent_loss.append(intent_loss)

        pbar.n = pbar.last_print_n = 0

    def score_function(engine):
        return engine.state.metrics['intent_acc']

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    to_save = {'model': model}
    checker = Checkpoint(to_save, DiskSaver('checkpoints', create_dir=True, require_empty=False),
                         n_saved=n_saved, filename_prefix='best', score_function=score_function, score_name='intent_acc',
                         global_step_transform=global_step_from_engine(trainer))
    stopper = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)

    valid_evaluator.add_event_handler(Events.COMPLETED, checker)
    valid_evaluator.add_event_handler(Events.COMPLETED, stopper)

    trainer.run(train_loader, max_epochs=epochs)
    
    def save_list(l, name):
        with open(name, 'wb') as f: 
            pickle.dump(l, f) 
    
    save_list(train_scope_acc,'train_scope_acc')
    save_list(train_scope_loss,'train_scope_loss')
    save_list(train_intent_acc,'train_intent_acc')
    save_list(train_intent_loss,'train_intent_loss')
    save_list(val_scope_acc,'val_scope_acc')
    save_list(val_scope_loss,'val_scope_loss')
    save_list(val_intent_acc,'val_intent_acc')
    save_list(val_intent_loss,'val_intent_loss')
    
    

if __name__ == "__main__":
    data_path ="dataset/"
    train_batch_size=32 #help="input batch size for training (default: 32)")
    val_batch_size=32 #help="input batch size for validation (default: 32)")
    epochs=21 #help="number of epochs to train (default: 10)")
    patience=10 #help="number of more epochs before early stopping")
    n_saved=1 # help="number of models saved")
    lr=4.00E-05 # help="learning rate")
    gradient_accumulation_steps=1 #help="Accumulate gradients on several steps")
    max_norm=1.0 # help="Clipping gradient norm")
    dropout=0.3 #help="Dropout rate for model")
    warmup_proportion=0.1 # help="Proportion of training to perform linear learning rate warmup for.")
    fp16=""# help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    pretrained="bert-base-uncased"# help="model/path of pretrained model")
    subspace='false' #help="use subspace layers")
    hierarchy='true'# help="use hierarchical structure")
    domain_first='false' #help="domain representation first")

    subspace = str2bool(subspace)
    hierarchy = str2bool(hierarchy)
    domain_first = str2bool(domain_first)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train()
