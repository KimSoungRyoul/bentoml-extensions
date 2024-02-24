FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04
USER root
RUN apt update && \
    apt install -y --no-install-recommends \
            software-properties-common \
            lsb-release  \
           build-essential \
            ca-certificates \
            cmake  \
           ninja-build   \
          git  \
           gnupg2  \
           python3      \
       python3-dev     \
        python3-distutils    \
         python3-pip     \
        python-is-python3     \
        wget

RUN apt install -y protobuf-compiler libprotobuf-dev --no-install-recommends
#RUN apt-get install -y         intel-oneapi-tbb         intel-oneapi-mkl         intel-oneapi-mkl-devel         intel-oneapi-tbb-devel         intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic
RUN pip install intel-optimization-for-horovod==0.28.1.1 torch==2.0.1+cpu torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu intel_extension_for_pytorch==2.0.100 oneccl-bind-pt==2.0.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

#RUN /opt/intel/oneapi/setvars.sh --force
RUN pip install numpy pandas tabulate transformers datasets

RUN git clone https://github.com/intel/light-model-transformer.git /light-model-transformer
WORKDIR /light-model-transformer/BERT
RUN mkdir build

WORKDIR /light-model-transformer/BERT/build/
RUN cmake .. -DBACKENDS="PT"
RUN cmake --build . -j 8
ENV BERT_OP_PT_LIB=/light-model-transformer/BERT/build/src/pytorch_op/libBertOpPT.so
ENV PYTHONPATH=/light-model-transformer/BERT/python
# python -c 'import transformers; transformers.BertModel.from_pretrained("bert-base-uncased")'
# cd tests/ipex/
# python benchmark.py -m bert-base-uncased --bert-op --quantization --batch-size=4 --run-time=10