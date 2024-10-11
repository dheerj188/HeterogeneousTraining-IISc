#!/bin/sh 
#SBATCH --job-name=serial_job_test    ## Job name 
#SBATCH --ntasks=1## Run on a single CPU can take upto 10 
#SBATCH --time=24:00:00               ## Time limit hrs:min:sec, its specific to queue being used 
#SBATCH --output=/raid/cdsdhe/outputs/serial_test_job.out  ## Standard output 
#SBATCH --error=/raid/cdsdhe/errors/serial_test_job.err   ## Error log 
#SBATCH --gres=gpu:4                  ## GPUs needed, should be same as selected queue GPUs 
#SBATCH --partition=q_1day-4G         ## Specific to queue being used, need to select from queues available 
#SBATCH --mem=500GB                     ## Memory for computation process can go up to 32GB pwd; hostname; date |tee result
#SBATCH --cpus-per-task=96

docker run -t \
	--gpus '"device=1,2,3,4"' \
	$USE_TTY \
	--name $SLURM_JOB_ID \
	--ipc=host --shm-size=2G \
	--user $(id -u $USER):$(id -g $USER) \
	--rm \
	-v /:/workspace \
	-v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v /etc/shadow:/etc/shadow \
	cdsdhe/caibase  \
	bash -c 'cd /workspace/raid/cdsdhe/ && ./run_cai.sh'
