



### Installation
#### create conda environment

```python
conda create -n ccnet python=3.9
conda activate ccnet
```

#### Install dependencies

```python
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start 
#### 1. Deblurring

##### Train

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2233 basicsr/train.py -opt options/train/NAFNet-width32.yml --launcher pytorch
```

##### Test

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2233 basicsr/test.py -opt options/test/NAFNet-width32.yml --launcher pytorch
```

#### 2. Corrosion Classification

##### Train

```
python main.py
```

##### visualize

```
python visualize.py
```

