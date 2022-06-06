# Ablation Experiments on DBLP Datasete
# Setting: batch=20, iteration=40
if [ ! -d "./experiment_result" ]; then
  mkdir ./experiment_result
fi

if [ ! -d "./experiment_result/ablation_experiment/" ]; then
  mkdir ./experiment_result/ablation_experiment/
fi

current_time=`date "+%N"`
echo "Running Ablation Experiments"
echo "Running nc-score"
python ./ActiveRGCN/main.py -d dblp --testing --gpu 0 --set NC --batch 20 --iteration 40 > "./experiment_result/ablation_experiment/dblp_al_${current_time}_nc.txt"
echo "Running cie-score"
python ./ActiveRGCN/main.py -d dblp --testing --gpu 0 --set CIE --batch 20 --iteration 40 > "./experiment_result/ablation_experiment/dblp_al_${current_time}_cie.txt"
echo "Running cid-score"
python ./ActiveRGCN/main.py -d dblp --testing --gpu 0 --set CID --batch 20 --iteration 40 > "./experiment_result/ablation_experiment/dblp_al_${current_time}_cid.txt"
echo "Running norm-score"
python ./ActiveRGCN/main.py -d dblp --testing --gpu 0 --set norm --batch 20 --iteration 40 > "./experiment_result/ablation_experiment/dblp_al_${current_time}_norm.txt"
echo "Running ori-score"
python ./ActiveRGCN/main.py -d dblp --testing --gpu 0 --set ori --batch 20 --iteration 40 > "./experiment_result/ablation_experiment/dblp_al_${current_time}_ori.txt"