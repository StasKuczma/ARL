FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    && apt-get clean


RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    matplotlib 

WORKDIR /workspace

RUN rm -rf /workspace/NVIDIA_Deep_Learning_Container_License.pdf /workspace/README.md /workspace/docker-examples /workspace/examples /workspace/tutorials

CMD ["/bin/bash"]