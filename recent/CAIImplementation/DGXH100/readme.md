DGX file contains the docker image builder and docker based launch script to train GPTs on DGX clusters. 

1) run.sh ==> main docker launch script (launches run_cai.sh)

2) run_cai.sh ==> launch script to configure deep learning and hardware configuration

3) dockerfile ==> contains base image to run colossal ai framework seamlessly. 
