This folder contains code, files, **progress presentations**, research outcomes and some important links for heterogeneous training research carried out at MARS lab, Indian Institute of Science. For queries contact: Dheemanth R Joshi (dheemanth@fsid-iisc.in)

**Main Objectives of the Project**
1) Utilize heterogeneous system properties (CPU compute and memory) to break the GPU memory wall to scale large scale LLMs and ViTs on resource constrained clusters
2) Parallelize CPU and GPU to avoid CPU to be the computational bottleneck
3) Create a unified, platform independent heterogeneous framework which integrates with traditional training pipelines to train deep learning models of unprecedented scale on limited resource clusters.

**Hardware/Software presently in focus**
1) Framework: Colossal AI --->DeepSpeed--->Advanced ZeRO + CPU offload (Patrick Integration See References.md [5])
2) Platform: Nvidia DGX-H100 cluster with Intel Xeon dual socket 54 core processor (2TB RAM), 8x Nvidia H100 GPUs (80GB HBM memory) 
3) Platform: IISc's IoE cluster Node 8
   ===> CPU -> Intel Xeon 32 core Processor with 192GB Main Memory / GPU-> 4x Nvidia Tesla V100 32GB GPUs

**Ongoing Work: Updated (25/9/24)**
1) DGX-H100 cluster successfully setup. running workloads with docker containers/images
2) Benchmarking model sizes 21B/41B on DGX cluster. (synthetic data).
3) Benchmarking 9.1B parameter model on ioE. (glue and c4 dataset) (c4--> 300GB)
4) Literature review and analysis of CoTrain[3] and Patrick Star[5].

**Planned Future Work**
1) Power Profiling of CAI++ frameworks. 

**Previous Update: Updated (10/9/24)**
1) Benchmarking Colossal AI framework as a base research benchmark. We are analyzing the possible bottlenecks and memory constraints in this framework
2) Framework specification:
   => Gemini plugin: (corresponding to Patrick Star dynamic CPU-GPU memory manager)
   => Optimizer: Hybrid Adam (heterogeneous parameter update feature based on memory placement)
4) Progress: Successfully benchmarked  parameters upto 9.1 billion parameters. Compute and memory trace of these experiments are being generated.
5) Inference:
   a) Management to avoid OOM: Dynamic memory manager avoids GPU OOM by dynamically offloading OS, P & G to CPU.
   b) Hybrid Adam: tries to jointly update weights by involving both CPU and GPU
   
   c) Observations:
   1)This scheme scales the number of parameters that can be trained on single node systems however, fails to utilize entire CPU           memory
   2) Kernel Launch overhead for FWD/BWD pass:it can be observed from profiler trace that there is significant amount of kernel launch overhead in FWD/BWD pass. Sharding/Communication kernel management overhead must be observed to analyze the share of its overehad.
  
**Future Work as per (10/9/24)**

1) Tracing on other nodes: Comparing tracing on Nvidia DGX cluster.
2) Compile Kernels: compile the model to fuse the kernel task graph to avoid independent kenrel launch resulting in less kernel overhead.  
      

