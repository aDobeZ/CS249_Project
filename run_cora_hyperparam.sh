: '
HyperParameter testing on Cora Dataset
BATCH * ITR = 3600
20 * 180
50 * 72
80 * 45
120 * 30
200 * 18
400 * 9
'
if [ ! -d "./experiment_result" ]; then
  mkdir ./experiment_result
fi

if [ ! -d "./experiment_result/hyperparam_experiment" ]; then
  mkdir ./experiment_result/hyperparam_experiment
fi

current_time=`date "+%N"`
for i in 20 50 80 120 200 400; do
    # echo "$i"
    itr=$((3600 / $i))
    # echo "$itr"
    python ./ActiveRGCN/main.py -d cora --testing --gpu 0 --set ori --batch $i --iteration $itr > "./experiment_result/hyperparam_experiment/cora_al_${current_time}_batch_${i}_itr_${itr}.txt"
done