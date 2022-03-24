import os
import torch
import shutil
import json

from .logger import Logger
from utils.utils import create_model_dir, create_metrics_dict


def save_checkpoint(model, filename, args, is_best, metrics, metric_to_optim):
    """
    Persist checkpoint to disk
    :param model:
    :param filename: Filename to persist model by
    :param args: training setup
    :param is_best: Whether model with best metric
    :param metrics: metrics obtained from evaluation
    :param metric_to_optim: metric to optimize, e.g. top 1 accuracy
    """

    result_text = f"avg_loss={metrics['loss'].get_avg()}," \
                   f" avg_{metric_to_optim}=" \
                   f"{metrics[metric_to_optim].get_avg()}"
    metrics_dict = create_metrics_dict(metrics)

    # if get_model_str_from_obj(model) == "DataParallel":
    #     state = model.module
    # else:
    #     state = model

    model_dir = create_model_dir(args)
    model_filename = os.path.join(model_dir, filename)
    result_filename = os.path.join(model_dir, 'results.txt')
    latest_filename = os.path.join(model_dir, 'latest.txt')

    # best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    # last_filename = os.path.join(model_dir, 'model_last.pth.tar')

    best_metric_filename = os.path.join(model_dir, 'best_metrics.json')
    last_metric_filename = os.path.join(model_dir, 'last_metrics.json')

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Logger.get().info("Saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result_text)
    # torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    save_dict_to_json(metrics_dict, last_metric_filename)
    if is_best:
        Logger.get().info("Found new best.")
        # shutil.copyfile(model_filename, best_filename)
        shutil.copyfile(last_metric_filename, best_metric_filename)
    # Logger.get().info("Copying to {}".format(last_filename))
    # shutil.copyfile(model_filename, last_filename)

    Logger.get().info("Removing redundant files")
    files_to_keep = ['model_last.pth.tar', 'model_best.pth.tar',
                     'results.txt', 'latest.txt', 'args.json',
                     'last_metrics.json', 'best_metrics.json']
    files_to_delete = [
        file for file in os.listdir(model_dir) if file not in files_to_keep]
    for f in files_to_delete:
        if not os.path.isdir(f):
            os.remove("{}/{}".format(model_dir, f))
    return


def load_checkpoint(model_dir, load_best=True):
    """
    Load model from checkpoint.
    :param model_dir: Directory to read the model from.
    :param load_best: Whether to read best or latest version of the model
    :return: The state dictionary of the model
    """

    if load_best:
        model_filename = os.path.join(model_dir, 'model_best.pth.tar')
        metric_filename = os.path.join(model_dir, 'best_metrics.json')
    else:
        model_filename = os.path.join(model_dir, 'model_last.pth.tar')
        metric_filename = os.path.join(model_dir, 'last_metrics.json')

    Logger.get().info("Loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    Logger.get().info("Loaded checkpoint '{}'".format(model_filename))
    # read checkpoint json
    with open(metric_filename, 'r') as f:
        metrics = json.load(f)
    return state, metrics['round']


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) if (isinstance(v, float) or isinstance(v, int)) else [
            float(e) for e in v] for k, v in d.items()}
        json.dump(d, f, indent=4)
