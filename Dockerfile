FROM continuumio/miniconda3

ADD spec-file.txt /tmp/spec-file.txt
ADD run_gmd.sh /run_gmd.sh

RUN chmod a+x /run_gmd.sh
RUN conda create --name gmd --file /tmp/spec-file.txt

RUN echo "source activate gmd" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN git clone https://github.com/SeanTBlack/FNLGMD.git