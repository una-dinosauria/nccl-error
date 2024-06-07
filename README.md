
on your dev machine

```bash
# Clone to RSC, get my branch
cd ~
git clone git@github.com:una-dinosauria/nccl-error.git
cd nccl-error
```

Launch the job
```bash
./launch_job.sh 1
```

This will launch a job on a single node, with all 8 gpus.
Each GPU is given to a task. DDP should work here, but you should see this error in the logs instead:

```
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error.
Last error:
nvmlDeviceGetHandleByPciBusId() failed: Not Found
    return dist._verify_params_across_processes(process_group, tensors, logger)
torch.distributed.DistBackendError: NCCL error in: /opt/conda/conda-bld/pytorch_1695392067780/work/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1331, unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.18.5
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error
```
