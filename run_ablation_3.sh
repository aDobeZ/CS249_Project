# Ablation Experiments on DBLP Datasete
# Setting: batch=20, iteration=40
echo "Running Ablation Experiments"
echo "Running nc-score"
python ./rgcn/main.py -d dblp --testing --gpu 0 --set NC --batch 20 --iteration 40 > ./ablation_experiment/dblp_al_nc.txt
echo "Running cie-score"
python ./rgcn/main.py -d dblp --testing --gpu 0 --set CIE --batch 20 --iteration 40 > ./ablation_experiment/dblp_al_cie.txt
echo "Running cid-score"
python ./rgcn/main.py -d dblp --testing --gpu 0 --set CID --batch 20 --iteration 40 > ./ablation_experiment/dblp_al_cid.txt
echo "Running norm-score"
python ./rgcn/main.py -d dblp --testing --gpu 0 --set norm --batch 20 --iteration 40 > ./ablation_experiment/dblp_al_norm.txt
echo "Running ori-score"
python ./rgcn/main.py -d dblp --testing --gpu 0 --set ori --batch 20 --iteration 40 > ./ablation_experiment/dblp_al_ori.txt