import argparse
import os

import datetime
import torch
import torch.distributed as dist
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Repro of NCCL issues ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
args = parser.parse_args()

best_acc1 = 0.


def main():
    slurm_procid = int(os.environ["SLURM_PROCID"])
    slurm_ntasks = int(os.environ["SLURM_NTASKS"])

    args.world_size = slurm_ntasks
    args.world_rank = slurm_procid

    ngpus_per_node = torch.cuda.device_count()
    
    # Simply call main_worker function
    main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    print(f"Initializing distributed process group: world_size={args.world_size}, world_rank={args.world_rank}")
    dist.init_process_group(backend="nccl",
                            world_size=args.world_size, 
                            rank=args.world_rank, 
                            timeout=datetime.timedelta(seconds=60))
    print(f"Distributed process group initialized at rank {args.world_rank}", flush=True)
    
    # create a model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # Make model DDP -- should fail here
    # torch.cuda.set_device(args.gpu)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)


if __name__ == '__main__':
    main()
