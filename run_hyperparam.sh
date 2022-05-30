: '
HyperParameter testing on Cora Dataset
BATCH * ITR
20 * 180
50 * 72
80 * 45
120 * 30
200 * 18
400 * 9
'
for i in 20 50 80 120 200 400; do
    # echo "$i"
    itr=$((3600 / $i))
    # echo "$itr"
    python ./rgcn/main.py -d cora --testing --gpu 0 --set ori --batch $i --iteration $itr > "./hyperparam_experiment/cora_al_batch_${i}_itr_${itr}.txt"
done