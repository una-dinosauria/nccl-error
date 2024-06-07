# After syncing everything, launch a job with this command

# Specify the number of nodes with argv[1]. Default value: 2
nodes=${1:-'2'}

SCENV=ava rsc_launcher launch --projects airstore_no_user_data_avatar_imagenet --end-script \
  'source /uca/conda-envs/activate-latest && cd /home/$USER/nccl-error && sbatch --nodes='"$nodes"' slurm.sh'
