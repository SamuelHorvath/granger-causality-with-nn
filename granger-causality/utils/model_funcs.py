import torch
from torch import nn
from torch.nn import DataParallel
import time

from sklearn import metrics

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.optim import SGD, Adam, RMSprop

import models
from utils.logger import Logger
from .utils import init_metrics_meter, log_epoch_info
from utils.checkpointing import load_checkpoint

CLIP_RNN_GRAD = 5


def get_training_elements(model_name, input_size, seq_len, resume_from, load_best, gpu):
    # Define the model
    model, current_round = initialise_model(
        model_name, input_size, seq_len, resume_from, load_best)

    model = model_to_device(model, gpu)

    criterion = nn.MSELoss()

    return model, criterion, current_round


def initialise_model(model_name, input_size, seq_len, resume_from=None, load_best=None):

    model = getattr(models, model_name)(input_size, seq_len)

    current_round = 0
    if resume_from:
        model, current_round = load_checkpoint(resume_from, load_best)

    return model, current_round


def model_to_device(model, device):
    if type(device) == list:  # if to allocate on more than one GPU
        model = model.to(device[0])
        model = DataParallel(model, device_ids=device)
    else:
        model = model.to(device)
    return model


def get_lr_scheduler(optimiser, total_epochs, method='static'):
    """
    Implement learning rate scheduler.
    :param optimiser: A reference to the optimiser being used
    :param total_epochs: The total number of epochs (from the args)
    :param method: The strategy to adjust the learning rate
    (multistep, cosine or static)
    :returns: scheduler on current step/epoch/policy
    """
    if method == 'cosine':
        return CosineAnnealingLR(optimiser, total_epochs)
    elif method == 'static':
        return MultiStepLR(optimiser, [total_epochs + 1])
    if method == 'cifar_1':
        return MultiStepLR(optimiser, [int(0.5 * total_epochs),
                                       int(0.75 * total_epochs)], gamma=0.1)
    if method == 'cifar_2':
        return MultiStepLR(optimiser, [int(0.3 * total_epochs),
                                       int(0.6 * total_epochs), int(0.8 * total_epochs)],
                           gamma=0.2)
    raise ValueError(f"{method} is not defined as scheduler name.")


def run_one_epoch(
        model, train_loader, criterion, optimiser,
        device, round, is_rnn=False, clip_grad=False, prox=False, lam=0.,
        print_freq=10):

    metrics_meter = init_metrics_meter(round)
    model.train()

    for i, (data, label) in enumerate(train_loader):
        start_ts = time.time()
        batch_size = data.shape[0]
        data, label = data.to(device), label.to(device)
        dataload_duration = time.time() - start_ts

        inference_duration = 0.
        backprop_duration = 0.

        optimiser.zero_grad(set_to_none=True)

        input, label = get_train_inputs(
            data, label, model, batch_size, device, is_rnn)
        inference_duration, backprop_duration, _, _ = \
            forward_backward(
                model, criterion, input, label, inference_duration,
                backprop_duration, batch_size, metrics_meter, is_rnn,
                prox, lam)
        if is_rnn or clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_RNN_GRAD)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        if i % print_freq == 0:
            log_epoch_info(
                Logger, i, len(train_loader), metrics_meter, dataload_duration,
                inference_duration, backprop_duration, train=True)

    return metrics_meter


def get_train_inputs(data, label, model, batch_size, device, is_rnn):
    if not is_rnn:
        input = (data,)
    else:
        hidden = model.init_hidden(batch_size, device)
        input = (data, hidden)
        # label = label.reshape(-1)
    return input, label


def evaluate_model(model, val_loader, criterion, device, round,
                   print_freq=10, metric_to_optim='loss', is_rnn=False, GC=None):
    metrics_meter = init_metrics_meter(round)
    if is_rnn:
        hidden = model.init_hidden(val_loader.batch_size, device)

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            batch_size = data.shape[0]
            start_ts = time.time()

            data = data.to(device)
            label = label.to(device)
            # if is_rnn:
            #     label = label.reshape(-1)

            dataload_duration = time.time() - start_ts
            if is_rnn:
                # output, hidden = model(data, hidden)
                output, _ = model(data, None)
            else:
                output = model(data)
            inference_duration = time.time() - (start_ts + dataload_duration)

            loss = compute_loss(criterion, output, label)
            update_metrics(metrics_meter, loss, batch_size)
            if i % print_freq == 0:
                log_epoch_info(
                    Logger, i, len(val_loader), metrics_meter,
                    dataload_duration, inference_duration,
                    backprop_duration=0., train=False)

    fpr, tpr, _ = metrics.roc_curve(GC.flatten(),  model.GC().flatten())
    auc_roc = metrics.auc(fpr, tpr)

    auc_pr = metrics.average_precision_score(GC.flatten(),  model.GC().flatten())

    metrics_meter['roc_auc'] = auc_roc
    metrics_meter['pr_auc'] = auc_pr
    metrics_meter['GC'] = model.GC().flatten()
    metrics_meter['GC_full'] = model.GC(ignore_lag=False).flatten()

    # Metric for avg/single model(s)
    Logger.get().info(f'{metric_to_optim}:'
                      f' {metrics_meter[metric_to_optim].get_avg()}'
                      f' ROC (AUC): {auc_roc:.4f},'
                      f' PR (AUC): {auc_pr:.4f}')

    return metrics_meter


def forward_backward(model, criterion, input, label, inference_duration,
                     backprop_duration, batch_size, metrics_meter, is_rnn,
                     prox, lam):
    start_ts = time.time()
    outputs = model(*input)

    if not is_rnn:
        hidden = None
        output = outputs
    else:
        output, hidden = outputs

    single_inference = time.time() - start_ts
    inference_duration += single_inference

    loss = compute_loss(criterion, output, label)
    if not prox:
        loss += model.regularize(lam=lam)
    loss.backward()
    backprop_duration += time.time() - (start_ts + single_inference)

    update_metrics(metrics_meter, loss, batch_size)
    return inference_duration, backprop_duration, output, hidden


def compute_loss(criterion, output, label):
    loss = criterion(output, label)
    return loss


def update_metrics(metrics_meter, loss, batch_size):
    metrics_meter['loss'].update(loss.item(), batch_size)


def get_optimiser(params_to_update, optimiser_name, lr, momentum, weight_decay):
    if optimiser_name == 'sgd':
        optimiser = SGD(params_to_update, lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
    elif optimiser_name == 'adam':
        optimiser = Adam(params_to_update, lr)
    elif optimiser_name == 'rmsprop':
        optimiser = RMSprop(params_to_update, lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    else:
        raise ValueError("optimiser not supported")

    return optimiser


@torch.no_grad()
def prox_step(model, scheduler, lam):
    model.prox(lam * scheduler.get_last_lr()[0])
