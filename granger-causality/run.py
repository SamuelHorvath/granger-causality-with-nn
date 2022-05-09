import sys
import os
import json
import torch
import numpy as np
import time

from opts import parse_args
from utils.logger import Logger
from data_funcs.data_loader import load_data
from utils.model_funcs import get_training_elements, evaluate_model, \
    get_lr_scheduler, get_optimiser, run_one_epoch, prox_step
from utils.checkpointing import save_checkpoint
from utils.utils import create_model_dir, create_metrics_dict, \
    metric_to_dict, init_metrics_meter, extend_metrics_dict

from models import RNN_MODELS

# os.environ["WANDB_API_KEY"] = 'INSERT KEY'
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


def main(args):
    # system setup
    global CUDA_SUPPORT

    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    logger = Logger()

    logger.debug(f"CLI args: {args}")

    if torch.cuda.device_count():
        CUDA_SUPPORT = True
    else:
        logger.warning('CUDA unsupported!!')
        CUDA_SUPPORT = False

    if not CUDA_SUPPORT:
        args.gpu = "cpu"

    if args.deterministic:
        # import torch.backends.cudnn as cudnn
        import os
        import random

        if CUDA_SUPPORT:
            # cudnn.deterministic = args.deterministic
            # cudnn.benchmark = not args.deterministic
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)

        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    logger.info(f"Model: {args.model}, Dataset:{args.dataset}")

    # In case of DataParallel for .to() to work
    args.device = args.gpu[0] if type(args.gpu) == list else args.gpu

    # Load data sets
    trainset, testset, GC = load_data(args.data_path, args.dataset, args.seq_len)

    # It should work, it seems problem is only locally on my machine
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             persistent_workers=True
                                             )
    args.input_size = next(iter(testloader))[0].shape[-1]

    if not args.evaluate:  # Training mode
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            persistent_workers=True
        )

        init_and_train_model(args, trainloader, testloader, GC)

    else:  # Evaluation mode
        model, criterion, round = get_training_elements(
            args.model_name, args.input_size, args.seq_len, args.resume_from,
            args.load_best, args.gpu)

        metrics = evaluate_model(
            model, testloader, criterion, args.device, round,
            print_freq=10, metric_to_optim=args.metric,
            is_rnn=args.model in RNN_MODELS, GC=GC)

        metrics_dict = create_metrics_dict(metrics)
        logger.info(f'Validation metrics: {metrics_dict}')


def init_and_train_model(args, trainloader, testloader, GC):
    full_metrics = init_metrics_meter()
    model_dir = create_model_dir(args)
    # don't train if setup already exists
    if os.path.isdir(model_dir):
        Logger.get().info(f"{model_dir} already exists.")
        Logger.get().info("Skipping this setup.")
        return
    # create model directory
    os.makedirs(model_dir, exist_ok=True)
    # init wandb tracking
    # wandb.init(
    #     project="insert_project_name", entity="insert_entity", config=vars(args),
    #     name=str(create_model_dir(args, lr=False)), reinit=True)
    # save used args as json to experiment directory
    with open(os.path.join(create_model_dir(args), 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    is_rnn = args.model in RNN_MODELS
    model, criterion, current_round = get_training_elements(
        args.model, args.input_size, args.seq_len, args.resume_from, args.load_best, args.gpu)

    optimiser = get_optimiser(
        model.parameters(), args.optimiser, args.lr,
        args.momentum, args.weight_decay)

    scheduler = get_lr_scheduler(
        optimiser, args.rounds, args.lr_type)

    metric_to_optim = args.metric
    best_metric = np.inf
    train_time_meter = 0

    for i in range(args.rounds):
        start = time.time()
        metrics_meter = run_one_epoch(
            model, trainloader, criterion, optimiser, args.device,
            current_round, is_rnn, args.clip_grad, args.prox, args.gc_penalty)
        # Prox step to better detect Granger Causality
        if args.prox:
            prox_step(model, scheduler, args.gc_penalty)
        extend_metrics_dict(
            full_metrics, metric_to_dict(metrics_meter, i+1, 'train'))
        # wandb.log(metric_to_dict(metrics_meter, i+1, 'train', False))
        train_time = time.time() - start
        train_time_meter += train_time
        # Track timings across epochs
        Logger.get().debug(f'Epoch train time: {train_time}')

        if i % args.eval_every == 0 or i == (args.rounds - 1):
            metrics = evaluate_model(
                model, testloader, criterion, args.device, current_round,
                print_freq=10, metric_to_optim=metric_to_optim, is_rnn=is_rnn,
                GC=GC)
            extend_metrics_dict(
                full_metrics, metric_to_dict(metrics, i+1, 'test'))
            # wandb.log(metric_to_dict(metrics, i+1, 'test', False))
            avg_metric = metrics[metric_to_optim].get_avg()
            # Save model checkpoint
            model_filename = (f"{args.model}_{args.run_id}_checkpoint"
                              f"_{current_round:0>2d}.pth.tar")
            is_best = avg_metric < best_metric
            save_checkpoint(model, model_filename, is_best=is_best, args=args,
                            metrics=metrics, metric_to_optim=metric_to_optim)
            if is_best:
                best_metric = avg_metric

            if np.isnan(metrics['loss'].get_avg()):
                Logger.get().info(
                    'NaN loss detected, aborting training procedure.')
                return

        Logger.get().info(
            f'Current lr: {scheduler.get_last_lr()}')
        scheduler.step()
        current_round += 1
        torch.cuda.empty_cache()

    Logger.get().debug(
        f'Average epoch train time: {train_time_meter / args.rounds}')
    #  store the run
    with open(os.path.join(
            create_model_dir(args), 'full_metrics.json'), 'w') as f:
        del full_metrics['train_GC']
        del full_metrics['test_GC']
        del full_metrics['train_GC_full']
        del full_metrics['test_GC_full']
        json.dump(full_metrics, f, indent=4)

    Logger.get().debug(f'GC model:{model.GC()}')
    Logger.get().debug(f'GC true:{GC}')


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() == 0
