import argparse
import time
from datetime import datetime
import os
import torch


def parse_args(args):
    parser = initialise_arg_parser(args, 'Granger Causality.')

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of epochs / rounds to run"
    )
    parser.add_argument(
        "--clip-grad",
        default=False,
        action='store_true',
        help="Whether to use gradient clipping"
    )
    parser.add_argument(
        "--prox",
        default=False,
        action='store_true',
        help="Whether to use prox step"
    )
    parser.add_argument(
        "--gc-penalty",
        type=float,
        default=1e-5,
        help="Group lasso penalty to detect Granger causality"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1024,
        help="Static batch size for local runs"
    )
    parser.add_argument(
        "-s", "--seq-len",
        type=int,
        default=5,
        help="Sequence length to consider for time series"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--lr-type',
        type=str,
        choices=['cosine', 'cifar_1', 'cifar_2', 'static'],
        default='cifar_1',
        help='Learning rate strategy (default: cifar_1)'
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        choices=['sgd', 'adam'],
        default='adam',
        help='Optimiser to use (default: SGD)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.,
        help='Momentum (default: 0.)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.,
        help='Weight decay (default: 0.)'
    )

    # MODEL and DATA PARAMETERS
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Define which dataset to load."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='loss',
        choices=["loss"],
        help="Define which metric to optimize."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Define which model to load"
    )

    # SETUP ARGUMENTS
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default='../check_points',
        help="Directory to persist run meta data_preprocess,"
             " e.g. best/last models."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from checkpoint."
    )
    parser.add_argument(
        "--load-best",
        default=False,
        action='store_true',
        help="Load best from checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/",
        help="Base root directory for the dataset."
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Define on which GPU to run the model"
             " (comma-separated for multiple). If -1, use CPU."
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        default=4,
        help="Num workers for dataset loading"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run deterministically for reproducibility."
    )
    parser.add_argument(
        "--manual-seed",
        type=int,
        default=123,
        help="Random seed to use."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="How often to do validation."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=str(time.time()),
        help="Identifier for the current job"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="INFO"
    )
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    os.makedirs("../logs/", exist_ok=True)
    parser.add_argument(
        "--logfile",
        type=str,
        default=f"../logs/log_{now}.txt"
    )

    # Evaluation mode, do not run training
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="Evaluation or Training mode")

    args = parser.parse_args()
    transform_gpu_args(args)

    return args


def get_available_gpus():
    """
    Get list of available gpus in the system
    """
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpus.append(torch.cuda.get_device_properties(i))


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(args, description=description)
    return parser


def transform_gpu_args(args):
    if args.gpu == "-1":
        args.gpu = "cpu"
    else:
        gpu_str_arg = args.gpu.split(',')
        if len(gpu_str_arg) > 1:
            args.gpu = sorted([int(card) for card in gpu_str_arg])
        else:
            args.gpu = f"cuda:{args.gpu}"
