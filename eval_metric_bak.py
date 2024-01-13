# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope
from rich import print


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate metric of the " "results saved in pkl format"
    )
    parser.add_argument(
        "--config",
        default="/ai/mnt/code/mmaction2/work_dirs_update_samples/c3d_sports1m-pretrained_8xb30-16x1x1-200e_AmTICIS-rgb/c3d_sports1m-pretrained_8xb30-16x1x1-200e_AmTICIS-rgb.py",
        help="Config of the model",
    )
    parser.add_argument("--pkl_results", default="NOT", help="Results in pickle format")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.pkl_results = glob.glob(
        os.path.join(os.path.dirname(args.config), "best_acc_top1_*.pkl")
    )[0]
    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get("default_scope", "mmaction"))

    data_samples = mmengine.load(args.pkl_results)

    evaluator = Evaluator(cfg.test_evaluator)
    eval_results = evaluator.offline_evaluate(data_samples)
    print(eval_results)


if __name__ == "__main__":
    main()
