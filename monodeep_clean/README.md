# stereoCNN
In development...

# Kitti Dataset
Most Indicated scenes list (without static scenes):

**City**: All except 0017, 0018, 0057, 0060, 0002, 0026

**Residential**: All

**Road**: All

**Campus**: Only 0038, 0045, 0047

**Person**: n/a

# Dataset Prepation Script

Ex: python3 dataset_preparation.py -s kittiRaw
    python3 dataset_preparation.py -s kitti2012
    python3 dataset_preparation.py -s kitti2015
    python3 dataset_preparation.py -s nyuDepth

# Restore Command Example
Ex: python3 stereo_cnn.py -i output/dataset_preparation/kittiRaw_road.pkl -r output/stereo_cnn/kittiRaw_road/2017-10-10_19-08-06/restore
    python3 stereo_cnn.py -i output/dataset_preparation/kittiRaw_city.pkl -r output/stereo_cnn/kittiRaw_city/2017-10-10_20-36-35/restore
