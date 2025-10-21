import os
import builtins
import datetime
import subprocess

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # kwargs['flush'] = True
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()



def init_distributed_mode(args):
    # if 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.world_size = int(os.environ['SLURM_NTASKS'])
        
    #     num_gpus = torch.cuda.device_count()
    #     args.gpu = args.rank % num_gpus
    #     args.local_rank = args.gpu

    #     node_list = os.environ['SLURM_NODELIST']
    #     print(f'Node list: {node_list}')
    #     addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')

    #     os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
    #     os.environ['MASTER_ADDR'] = addr
    #     os.environ['WORLD_SIZE'] = str(args.world_size)
    #     os.environ['LOCAL_RANK'] = str(args.gpu)
    #     os.environ['RANK'] = str(args.rank)
    # elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = args.gpu
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank,
                            timeout=datetime.timedelta(0, 7200))
    dist.barrier()
    setup_for_distributed(args.rank == 0)

    