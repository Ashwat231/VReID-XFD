from utils.logger import setup_logger
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage_dat_and_prompt
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss.make_loss_video import make_loss
from processor.processor_videoreid_stage1 import do_train_stage1
from processor.processor_videoreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--stage1weight", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.stage1weight = args.stage1weight
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    (train_loader_stage2, train_loader_stage1,
     query_loader, gallery_loader,
     _, _,
     num_classes, num_query, num_camera) = make_dataloader(cfg)

    # TODO
    model = make_model(cfg, num_class=num_classes, camera_num=num_camera, view_num=0)
#ADDED START--------------------------------------------------------------------------
    for name, param in model.named_parameters():
        lname = name.lower()
        if("prompt" in lname or "adapter" in lname or "dat" in lname or "pbp" in lname or "qtw" in lname or "vcah" in lname):
            param.requires_grad = True
        elif("image_encoder" in lname and ("resblocks.10" in lname or "resblocks.11" in lname)):
            param.requires_grad = True
        elif("classifier" in lname or "bottleneck" in lname):
            param.requires_grad = True
        else:
            param.requires_grad = False
#ADDED END--------------------------------------------------------------------------

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # optimizer_1stage = make_optimizer_1stage(cfg, model)  # CHANGED
    # scheduler_1stage = CosineAnnealingLR(optimizer_1stage, T_max=cfg.SOLVER.STAGE1.MAX_EPOCHS, eta_min=cfg.SOLVER.STAGE1.LR_MIN)  # CHANGED

    base_lr_s1 = cfg.SOLVER.STAGE1.BASE_LR      # CHANGED
    backbone_lr_s1 = base_lr_s1 * 0.1           # CHANGED

    backbone_params_s1 = []                     # CHANGED
    other_params_s1 = []                        # CHANGED

    for name, p in model.named_parameters():    # CHANGED
        if not p.requires_grad:                 # CHANGED
            continue                            # CHANGED
        lname = name.lower()                    # CHANGED
        if ("image_encoder" in lname and        # CHANGED
            ("resblocks.10" in lname or "resblocks.11" in lname)):  # CHANGED
            backbone_params_s1.append(p)        # CHANGED
        else:                                   # CHANGED
            other_params_s1.append(p)           # CHANGED

    optimizer_1stage = torch.optim.Adam(        # CHANGED
        [                                      # CHANGED
            {"params": backbone_params_s1, "lr": backbone_lr_s1},  # CHANGED
            {"params": other_params_s1, "lr": base_lr_s1},        # CHANGED
        ],                                      # CHANGED
        weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY  # CHANGED
    )                                           # CHANGED

    scheduler_1stage = CosineAnnealingLR(       # CHANGED
        optimizer_1stage,                       # CHANGED
        T_max=cfg.SOLVER.STAGE1.MAX_EPOCHS,     # CHANGED
        eta_min=cfg.SOLVER.STAGE1.LR_MIN        # CHANGED
    )                                           # CHANGED

    if cfg.stage1weight != "":
        logger.info("skip stage one")
        weight_dict = torch.load(cfg.stage1weight)
        model.load_state_dict(weight_dict, strict=False)
    else:
        do_train_stage1(
            cfg,
            model,
            train_loader_stage1,
            optimizer_1stage,
            scheduler_1stage,
            args.local_rank
        )

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage_dat_and_prompt(cfg, model, center_criterion)
    scheduler_2stage = CosineAnnealingLR(optimizer_2stage, T_max = cfg.SOLVER.STAGE2.MAX_EPOCHS, eta_min=cfg.SOLVER.STAGE1.LR_MIN)

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        query_loader,
        gallery_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query,
        args.local_rank,
        scheduler_1stage
    )
