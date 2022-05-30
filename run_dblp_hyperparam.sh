: '
HyperParameter testing on DBLP Dataset
BATCH * ITR = 800
20 * 40
50 * 16
80 * 10
100 * 8
160 * 5
200 * 4
'
for i in 20 50 80 100 160 200; do
    # echo "$i"
    itr=$((800 / $i))
    # echo "$itr"
    python ./rgcn/main.py -d dblp --testing --gpu 0 --set ori --batch $i --iteration $itr > "./hyperparam_experiment/dblp_al_batch_${i}_itr_${itr}.txt"
done