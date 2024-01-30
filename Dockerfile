#55：nvidia/cuda:11.7.0-runtime-ubuntu22.04
#53：nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

#set rootpasswd
RUN echo "root:root" | chpasswd
#config ssh
RUN apt-get update && apt-get install -y openssh-server tmux vim \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "Port 22" >> /etc/ssh/sshd_config
RUN mkdir /var/run/sshd
EXPOSE 22 6006
CMD [ "/usr/sbin/sshd","-D" ]
#miniconda
ENV HOME "/root"
ENV CONDA_DIR "${HOME}/miniconda"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u \
    && "${CONDA_DIR}/bin/conda" init bash \
    && rm -f /tmp/miniconda3.sh \
    && echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"
SHELL ["/bin/bash", "--login", "-c"]
#pip install
COPY . /root/hifigan
RUN conda create --name test python=3.8
RUN conda activate test && pip install -r requirements.txt -i https://pypi.douban.com/simple/
RUN echo "service ssh restart" >> /root/.bashrc && \
    echo "conda activate hifigan" >> /root/.bashrc