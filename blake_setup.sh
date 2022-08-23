te -n rstenv -c conda-forge -c ets python=3.8 rstfinder
conda activate rstenv
pip install python-zpar
export NLTK_DATA="$HOME/nltk_data"
python -m nltk.downloader maxent_treebank_pos_tagger punkt

mkdir rstfinder
cd rstfinder

mkdir corpora
cd corpora
cp /iesl/data/ldc/LDC2002T07.tgz ./ # RST Discourse Treebank
tar -zxvf LDC2002T07.tgz
cp /iesl/data/ldc/LDC99T42.tgz ./ # Penn Treebank
tar -zxvf LDC99T42.tgz
cd ../

convert_rst_discourse_tb corpora/rst_discourse_treebank corpora/treebank_3
make_traindev_split
extract_segmentation_features rst_discourse_tb_edus_TRAINING_TRAIN.json
rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv

extract_segmentation_features rst_discourse_tb_edus_TRAINING_DEV.json
rst_discourse_tb_edus_features_TRAINING_DEV.tsv

tune_segmentation_model rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv
rst_discourse_tb_edus_features_TRAINING_DEV.tsv segmentation_model

tune_rst_parser rst_discourse_tb_edus_TRAINING_TRAIN.json
rst_discourse_tb_edus_TRAINING_DEV.json rst_parsing_model

rst_eval rst_discourse_tb_edus_TRAINING_DEV.json -p rst_parsing_model.C1.0
--use_gold_syntax

mkdir zpar-models
cd zpar-models
wget
https://github.com/frcchang/zpar/releases/download/v0.7.5/english-models.zip
unzip english-models.zip
mv english-models/* ./
rmdir english-models
cd ../

export NLTK_DATA="$HOME/nltk_data"
export ZPAR_MODEL_DIR="`pwd`/zpar-models"

rst_parse -g segmentation_model.C1.0 -p rst_parsing_model.C1.0 document.txt >
output.json

visualize_rst_tree output.json tree.html --embed_d3js
