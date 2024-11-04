# Use the official NVIDIA CUDA runtime image with cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Miniconda
RUN apt-get update && apt-get install -y wget git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash

# Set up the environment
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /usr/src/app
COPY env.yml .
RUN conda env create -f env.yml

# # Activate the conda environment by default
RUN echo "source activate evaluator" > ~/.bashrc
ENV CONDA_DEFAULT_ENV=evaluator
ENV PATH /opt/conda/envs/evaluator/bin:$PATH

# Copy the application code
COPY . /usr/src/app 
# #since it is currently a private repository you need the GITHUB ACCES TOKEN
# RUN rm -rf /usr/src/app/* &&  git clone https://LeJudith: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@github.com/LeJudith/medical-report-processor.git /usr/src/app

RUN ls -la /usr/src/app

#activate conda environment
RUN conda run --no-capture-output -n evaluator python 

# Set the environment variable to avoid bufferng
ENV PYTHONUNBUFFERED=1

# Set the entry point to use the conda environment and run the script

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "evaluator"]

#CMD to allow for different commands to be passed when running the container, prevents that the container exits immediately once started
CMD ["bash"]