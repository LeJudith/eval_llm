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

RUN ls -la /usr/src/app

#activate conda environment
RUN conda run --no-capture-output -n evaluator python 
#RUN conda run -n evaluator python3 -m spacy download en_core_web_sm

# Set the environment variable to avoid bufferng
ENV PYTHONUNBUFFERED=1
#set Pythonpath
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/eval_llm/"
ENV HF_HOME=/root/.cache/

# to run docker interactively
EXPOSE 22 6006 8888

# Set the entry point to use the conda environment and run the script
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "evaluator","python3", "src/evaluate.py"]

#CMD to allow for different commands to be passed when running the container, prevents that the container exits immediately once started #CMD ["--help"]
#CMD ["bash"]
CMD ["--help"]