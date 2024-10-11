# Heterogeneous Training 
sharable code and reports of my IISc Research on heterogeneous training of large-scale DNNs on CPU-GPU platforms.

Code is available in the "CodeBase" folder. Most of the code is implemented pytorch's distributed api and multiprocessing environment. for any queries, write to dheemanth@fsid-iisc.in 

Most recent benchmarks on heterogeneous memory management and 41B GPT training can be found in the "recent" folder. Our team is presently focussed on deepspeed and colossal ai. 

# Main Objectives of the Project

Utilize heterogeneous system properties (CPU compute and memory) to break the GPU memory wall to scale large scale LLMs and ViTs on resource constrained clusters
Parallelize CPU and GPU to avoid CPU to be the computational bottleneck
Create a unified, platform independent heterogeneous framework which integrates with traditional training pipelines to train deep learning models of unprecedented scale on limited resource clusters.
Hardware/Software presently in focus

Framework: Colossal AI --->DeepSpeed--->Advanced ZeRO + CPU offload (Patrick Integration See References.md [5])
Platform: Nvidia DGX-H100 cluster with Intel Xeon dual socket 54 core processor (2TB RAM), 8x Nvidia H100 GPUs (80GB HBM memory)
Platform: IISc's IoE cluster Node 8 ===> CPU -> Intel Xeon 32 core Processor with 192GB Main Memory / GPU-> 4x Nvidia Tesla V100 32GB GPUs
