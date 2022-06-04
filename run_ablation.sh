# Ablation Experiments on Cora Datasete
# Setting: batch=80, iteration=40
echo "Running Ablation Experiments"
echo "Running nc-score"
python ./rgcn/main.py -d cora --testing --gpu 0 --set NC --batch 80 --iteration 40 > ./ablation_experiment/cora_al_nc.txt
echo "Running cie-score"
python ./rgcn/main.py -d cora --testing --gpu 0 --set CIE --batch 80 --iteration 40 > ./ablation_experiment/cora_al_cie.txt
echo "Running cid-score"
python ./rgcn/main.py -d cora --testing --gpu 0 --set CID --batch 80 --iteration 40 > ./ablation_experiment/cora_al_cid.txt
echo "Running norm-score"
python ./rgcn/main.py -d cora --testing --gpu 0 --set norm --batch 80 --iteration 40 > ./ablation_experiment/cora_al_norm.txt
echo "Running ori-score"
python ./rgcn/main.py -d cora --testing --gpu 0 --set ori --batch 80 --iteration 40 > ./ablation_experiment/cora_al_ori.txt