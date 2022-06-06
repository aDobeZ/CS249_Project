# CS249_Project

### Group Members:
Xinyu Zhao, Hang Zhang, Haowei Jiang, Nuocheng Pan

### Repository Components:
- code
- [final slides](https://github.com/aDobeZ/CS249_Project/blob/main/249%20Final%20Project%20Presentation%20Slide.pptx)
- report
- [dataset](https://github.com/aDobeZ/CS249_Project/tree/main/data)

### Environment Requirement
- Tensorflow 1.8.0 (for ActiveHNE)

- python 3.6

- pytorch 

- dgl(for ActiveRGCN)  

Installation command:

```shell
pip install -r requirement.txt
```

### ActiveRGCN and RGCN comparison experiment
Method 1: 

```shell
./run_test.sh
```

Method 2: 

```shell
python ./ActiveRGCN/main.py -d dataset_name --testing --gpu 0 --iteration iteration_num --batch batch_num
```

dataset_name options include cora, movielens, dblp. When choosing iteration_num and batch_num, please make sure that iteration_num * batch_num < train_data size (3911 for cora, 918 for movielens, 1014 for dblp). The experiment result will be auto-saved in experiment_result/exp_data_{timestamp} directory experiment log will be auto-saved in hyperparam_experiment directory.  

### Ablation Experiments
command: 

```shell
./run_<dataset>_ablation.sh  
```

--set option in the shell script chooses the reward function for each experiment. Specifially, `--set norm` is the same-weight setting where each reward is assigned a 1/3 constant weight through all iterations. `--set ori` uses the original multi-armed bendit setting. `--set NC`, `--set CIE`, and `--set CID` only choose one reward score and its corresponding weight in the active select strategy.

The experiment result will be auto-saved in experiment_result/exp_data_{timestamp} directory, experiment log will be auto-saved in ablation_experiment directory.

### ActiveHNE Experiment
command: 

```shell
python ./ActiveHNE/ActiveHNE.py
```