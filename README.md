# CS249_Project

### ActiveRGCN and RGCN comparison experiment
Method 1: 

```shell
./run_test.sh
```

Method 2: 

```shell
python ./ActiveRGCN/main.py -d dataset_name --testing --gpu 0 --iteration iteration_num --batch batch_num
```

dataset_name includes cora, movielens, dblp  
notice iteration_num * batch_num < train_data size (3911 for cora, 918 for movielens, 1014 for dblp)  
The experiment result will be auto-saved in experiment_result/exp_data_{timestamp} directory, experiment log will be auto-saved in hyperparam_experiment directory  

### ActiveRGCN and RGCN comparison experiment
command: 

```shell
./run_ablation.sh  
```

--score represents the weight of reward function  
The experiment result will be auto-saved in experiment_result/exp_data_{timestamp} directory, experiment log will be auto-saved in ablation_experiment directory  

### ActiveHNE Experiment
command: 

```shell
python ./ActiveHNE/ActiveHNE.py
```

### Environment Requirement
- Tensorflow 1.8.0 (for ActiveHNE)

- python 3.6

- pytorch 

- dgl(for ActiveRGCN)  

Installation:

```shell
pip install -r requirement.txt
```

