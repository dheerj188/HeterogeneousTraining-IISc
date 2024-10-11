**File Breakdown and Description**

1) train_cai_gpt.py: python file which contains the training routine and profile hooks.

2) train_cai_gpt_glue.py: python file which contains profiling and training script of training GPT-2 with GLUE/c4 dataset

3) commons: contains dataloading/naive profiling and some basic utils for framework setup.

4) runcaigpt.sh: launch script to distribute training workload accross 4 processes (4 GPUs) to run gemini + hybrid Adam.

**Code Function**

The script launches a set of mentioned processes on a single node. It performs ZeRO based DP on each data parallel process. The configuration and runtime of ZeRO class is decided by the user. Static/dynamic runtimes can be configured. Parameters can dynamically sharded accross processes. For more info visit: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/gemini_plugin.py

**Results**

Latest results generated and inference is added in the "Results" directory. 

