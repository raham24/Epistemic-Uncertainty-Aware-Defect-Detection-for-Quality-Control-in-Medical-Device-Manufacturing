#!/bin/bash
# Allocate a compute node and run an sshd on it, so VS Code (Remote-SSH) can
# connect straight to the node and run a notebook there with the `paper` env.
# Adapted from the Star HPC "Jupyter on a compute node from VS Code" docs.
#
#   mkdir -p cluster/logs
#   sbatch cluster/tunnel.slurm          # then wait until: squeue --me  shows state R
#
# Prereqs (one-time, see the setup notes):
#   - an SSH key at ~/.ssh/id_ed25519 on the cluster (host key for this sshd)
#   - your public key in ~/.ssh/authorized_keys on the cluster (shared home)
#   - the matching local ~/.ssh/config entries for `login2` and `hpcx`
#SBATCH --job-name=tunnel
#SBATCH --output=cluster/logs/tunnel.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# For interactive TRAINING on a GPU node, add (types: A30 / A100 / H100):
#   #SBATCH --gres=gpu:A30:1
# The analysis notebooks are light, so CPU (no gres) is the sensible default.

# pick a free port on the node and stash it in the job's Comment, so the local
# ProxyCommand can discover (node, port) via `squeue ... -O NodeList,Comment`
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
scontrol update JobId="$SLURM_JOB_ID" Comment="$PORT"

echo "tunnel up on node $(hostname) port $PORT"
# host key for this throwaway sshd = your account key (per the Star HPC docs)
HOSTKEY="${HOME}/.ssh/id_ed25519"
exec /usr/sbin/sshd -D -p "${PORT}" -f /dev/null -h "${HOSTKEY}"
