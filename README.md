# bentoml-extensions [![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org) ![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

#### plan: release bentomlx 0.1.0  2024.05
#### support bentoml 1.1.x (Runner) & bentoml 1.2.x (Distributed Service) 

* FeatureStore Runner InnerService [redis, aerospike, elasticsearch],
* optimized cpu inference [ipex, ovms]


## QuickStart

~~~shell
pip install "bentomlx[redis]"
pip install "bentomlx[aerospike]"
pip install "intel-extension-for-pytorch bentomlx"

~~~

todo ...

~~~
bentomlx build -f bentofile.yaml
~~~

~~~yaml
# bentofile.yaml
docker:
    distro: debian
    python_version: "3.8.12"
    oneapi_version: xxx # <-- need to support of the BentoML, to read bentofile.yaml and override bentoml build command
    #cuda_version: "11.6.2"
    system_packages:
      - ...

~~~


### FeatureStore Runner (bentoml<=1.1.x)

...

### FeatureStore Inner(Distributed)Service (bentoml>=1.2.0)

...



## CPU Optimized Runner
  * `bentomlx[ipex]`
  * `bentomlx[ovms]` `like a bentoml[triton]`


## Post(Runtime) Model Compression (oneapi nncl)
  * post quant ?
  * ...


![스크린샷 2023-11-27 오후 3 18 18](https://github.com/KimSoungRyoul/bentoml-extensions/assets/24240623/8b922a8f-99e6-4d69-a713-a03f3f7b0d27)




~~~
❯ python fibo_main.py # mypyc
0.17745399475097656
0.1755237579345703
0.17790436744689941
0.18230915069580078

❯ python fibo_main.py # 3.11.7
1.2891952991485596
1.2943885326385498
1.2915637493133545
1.305750846862793

❯ pyenv global cinder-3.10-dev
❯ PYTHONJIT=1 python fibo_main.py
2.9099485874176025
2.918196678161621
2.929981231689453
2.9137821197509766

❯ pyenv global pypy3.10-7.3.15
❯ PYTHONJIT=1 python fibo_main.py
0.8286490440368652
0.8387455940246582
0.8492231369018555
0.84218430519104


~~~






#### BertOperator

#### Batch-Size 1

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-large-uncased --warmup-time 5 --run-time 20
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-large-uncased --warmup-time 5 --run-time 20
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
|    | Model              | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:-------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-large-uncased | False  | False     | False          | False      |            1 |       128 |                    0.878 | 1139.490 ms    |
~~~



~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-large-uncased --bert-op --warmup-time 5 --run-time 20 -q
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-large-uncased --bert-op --warmup-time 5 --run-time 20 -q
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
|    | Model              | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:-------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-large-uncased | False  | True      | True           | False      |            1 |       128 |                    6.124 | 163.285 ms     |
~~~


#### Batch-Size 10

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-large-uncased --warmup-time 5 --run-time 20 --batch-size 10
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-large-uncased --warmup-time 5 --run-time 20 --batch-size 10
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
|    | Model              | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:-------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-large-uncased | False  | False     | False          | False      |           10 |       128 |                    1.588 | 6296.495 ms    |

~~~


~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-large-uncased --bert-op  --warmup-time 5 --run-time 20 --batch-size 10 --quant
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-large-uncased --bert-op --warmup-time 5 --run-time 20 --batch-size 10 --quant
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
|    | Model              | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:-------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-large-uncased | False  | True      | True           | False      |           10 |       128 |                    5.959 | 1678.104 ms    |

~~~

### bert-base-uncased

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
config.json: 100%|██████████| 570/570 [00:00<00:00, 170kB/s]
model.safetensors: 100%|██████████| 440M/440M [00:12<00:00, 36.2MB/s]
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | False     | False          | False      |            1 |       128 |                    3.832 | 260.979 ms     |

~~~


~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op --quant
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op --quant
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
config.json: 100%|██████████| 570/570 [00:00<00:00, 168kB/s]
model.safetensors: 100%|██████████| 440M/440M [00:11<00:00, 37.0MB/s] 
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |            1 |       128 |                   16.622 | 60.160 ms      |

~~~


### bert-base-uncased (batch-size 10)

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op --quant --batch-size 10
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op --quant --batch-size 10
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
config.json: 100%|██████████| 570/570 [00:00<00:00, 172kB/s]
model.safetensors: 100%|██████████| 440M/440M [00:12<00:00, 35.2MB/s]
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |           10 |       128 |                   23.923 | 418.015 ms     |

~~~


### diff ( origin bert, ipex bert, bert operator)

~~~
---------------- BatchSize 1 ------------
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | False     | False          | False      |            1 |       128 |                     4.89 | 204.520 ms     |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | True   | False     | False          | False      |            1 |       128 |                    5.243 | 190.739 ms     |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | False          | False      |            1 |       128 |                     5.88 | 170.077 ms     |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |            1 |       128 |                   15.444 | 64.752 ms      |
~~~


~~~
---------------- BatchSize 10 ------------
|    | Model              | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:-------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-large-uncased | False  | False     | False          | False      |           10 |       128 |                    1.588 | 6296.495 ms    |
|---:|:-------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-large-uncased | False  | True      | True           | False      |           10 |       128 |                    5.959 | 1678.104 ms    |
~~~


~~~
-------------- BatchSize 20 -------------
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | False     | True           | False      |           20 |       128 |                    5.441 | 3675.675 ms    |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |           20 |       128 |                   16.334 | 1224.473 ms    |
~~~


~~~
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | False     | True           | False      |           10 |       128 |                    5.727 | 1746.223 ms    |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |           10 |       128 |                   17.384 | 575.239 ms     |

~~~



~~~
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | False     | True           | False      |          100 |       128 |                     5.15 | 19417.498 ms   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |          100 |       128 |                   18.638 | 5365.511 ms    |
~~~



#### origin bert

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
config.json: 100%|██████████| 570/570 [00:00<00:00, 169kB/s]
model.safetensors: 100%|██████████| 440M/440M [00:12<00:00, 36.2MB/s] 
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | False     | False          | False      |            1 |       128 |                     4.89 | 204.520 ms     |
~~~

#### ipex optimized bert

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --ipex
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --ipex
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
config.json: 100%|██████████| 570/570 [00:00<00:00, 175kB/s]
model.safetensors: 100%|██████████| 440M/440M [00:11<00:00, 38.7MB/s][W LegacyTypeDispatch.h:74] Warning: AutoNonVariableTypeMode is deprecated and will be removed in 1.10 release. For kernel implementations please use AutoDispatchBelowADInplaceOrView instead, If you are looking for a user facing API to enable running your inference-only workload, please use c10::InferenceMode. Using AutoDispatchBelowADInplaceOrView in user code is under risk of producing silent wrong result in some edge cases. See Note [AutoDispatchBelowAutograd] for more details. (function operator())

/usr/local/lib/python3.8/dist-packages/intel_extension_for_pytorch/frontend.py:396: UserWarning: Conv BatchNorm folding failed during the optimize process.
  warnings.warn("Conv BatchNorm folding failed during the optimize process.")
/usr/local/lib/python3.8/dist-packages/intel_extension_for_pytorch/frontend.py:401: UserWarning: Linear BatchNorm folding failed during the optimize process.
  warnings.warn("Linear BatchNorm folding failed during the optimize process.")
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | True   | False     | False          | False      |            1 |       128 |                    5.243 | 190.739 ms     |

~~~


#### bert operator
~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
config.json: 100%|██████████| 570/570 [00:00<00:00, 187kB/s]
model.safetensors: 100%|██████████| 440M/440M [00:12<00:00, 35.4MB/s] 
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | False          | False      |            1 |       128 |                     5.88 | 170.077 ms     |
~~~


#### bert operator (with quant)

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op --quant
 
:: initializing oneAPI environment ...
   entrypoint.sh: BASH_VERSION = 5.0.17(1)-release
   args: Using "$@" for setvars.sh arguments: numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op --quant
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: mkl -- latest
:: tbb -- latest
:: oneAPI environment initialized ::
 
config.json: 100%|██████████| 570/570 [00:00<00:00, 159kB/s]
model.safetensors: 100%|██████████| 440M/440M [00:11<00:00, 39.1MB/s] 
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |            1 |       128 |                   15.444 | 64.752 ms      |

~~~




#### origin bert VS bert operator (with quant, batch-size 20)

*  batch-size=1 : 성능차이가 10~20% 차이이지만
*  batch-size=20 : 3배 정도 성능 차이난다. 
* `DNNL_CPU_RUNTIME=TBB|OMP` 는 큰 차이를 확인 못함 이론상 tbb는 thread num이 늘어나도 성능저하 없는게 특징 

~~~
❯ sudo docker run --rm --privileged bert-op-pytorch-demo-oneapi-tbb-onednn-v34pc numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --quant --batch-size 20
   config.json: 100%|██████████| 570/570 [00:00<00:00, 5.15MB/s]
model.safetensors: 100%|██████████| 440M/440M [00:07<00:00, 61.7MB/s] 
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | False     | True           | False      |           20 |       128 |                    5.441 | 3675.675 ms    |

❯ sudo docker run --rm --privileged bert-op-pytorch-demo-oneapi-tbb-onednn-v34pc numactl --all -- -m bert-base-uncased --warmup-time 5 --run-time 20 --bert-op --quant --batch-size 20
config.json: 100%|██████████| 570/570 [00:00<00:00, 5.25MB/s]
model.safetensors: 100%|██████████| 440M/440M [00:06<00:00, 63.2MB/s] 
|    | Model             | IPEX   | BERT Op   | Quantization   | BFloat16   |   Batch Size |   Seq Len |   Throughput [samples/s] | Latency [ms]   |
|---:|:------------------|:-------|:----------|:---------------|:-----------|-------------:|----------:|-------------------------:|:---------------|
|  0 | bert-base-uncased | False  | True      | True           | False      |           20 |       128 |                   16.334 | 1224.473 ms    |

~~~

pip install  --index-url https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple

pip install dpnp numba-dpex dpctl intel-optimization-for-horovod==0.28.1.1 torch==2.0.1 torchvision==0.15.2 --extra-index-url=https://download.pytorch.org/whl/cpu intel_extension_for_pytorch==2.0.100 oneccl-bind-pt==2.0.0 --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
