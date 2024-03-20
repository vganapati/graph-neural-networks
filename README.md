# graph-neural-networks

### Setup on NERSC Perlmutter:

```
cd /global/cfs/cdirs/m3562/users/vidyagan
git clone https://github.com/vganapati/graph-neural-networks.git
module load conda

module load pytorch/2.1.0-cu12
conda create --prefix /global/common/software/m3562/gnn
conda activate /global/common/software/m3562/gnn
```

Test install:
```
cd graph-neural-networks
python intro.py
```

Deactivate environment:

```
conda deactivate
```

### Startup after setup:

```
cd /global/cfs/cdirs/m3562/users/vidyagan/graph-neural-networks
module load conda
module load pytorch/2.1.0-cu12
conda activate /global/common/software/m3562/gnn
```

To start an interactive session:
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m3562
```
