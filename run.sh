python 1_feature_leak.py
python 1_sample_and_feature.py -s 23 -e 24 tmp/CV_train_2324_top20_addfea.feather
python 1_sample_and_feature.py -s 25 -e 28 tmp/test_sample_feature2528_addfea.feather
python 1_sample_and_feature.py -s 29 -e 32 tmp/test_sample_feature2932_addfea.feather

python 3_lgb_v9.py