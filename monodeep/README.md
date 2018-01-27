# Monodeep
In development...

# Kitti Dataset
Most Indicated scenes list (without static scenes):

**City**: All except 0017, 0018, 0057, 0060, 0002, 0026

**Residential**: All

**Road**: All

**Campus**: Only 0038, 0045, 0047

**Person**: n/a

# Training
Ex: ./monodeep.py -m train -i /home/olorin/Documents/nicolas/tensorflow/tese/dataset_preparation/output/kittiraw_campus.pkl --max_steps 300 -t			(Deprecated)
    ./monodeep.py -m train -i /home/olorin/Documents/nicolas/tensorflow/tese/dataset_preparation/output/kittiraw_residential_continuous.pkl --max_steps 300 -t	(Deprecated)
	
    ./monodeep.py -m train -s kitti2012 --max_steps 1000 -t -d 0.5
    ./monodeep.py -m train -s kitti2015 --max_steps 1000 -t -d 0.5
    ./monodeep.py -m train -s kittiraw_campus --max_steps 1000 -t -d 0.5

# Testing
Ex: ./monodeep.py -m test -i /home/olorin/Documents/nicolas/tensorflow/tese/dataset_preparation/output/kittiraw_campus.pkl --max_steps 300 -t	(Deprecated)
    ./monodeep.py -m test -s kitti2012 -r output/monodeep/2018-01-27_14-20-07/restore/

# Dataset Prepation Script

Ex: python3 dataset_preparation.py -s kittiRaw
    python3 dataset_preparation.py -s kitti2012
    python3 dataset_preparation.py -s kitti2015
    python3 dataset_preparation.py -s nyuDepth

# Restore Command Example
Ex: python3 stereo_cnn.py -i output/dataset_preparation/kittiRaw_road.pkl -r output/stereo_cnn/kittiRaw_road/2017-10-10_19-08-06/restore
    python3 stereo_cnn.py -i output/dataset_preparation/kittiRaw_city.pkl -r output/stereo_cnn/kittiRaw_city/2017-10-10_20-36-35/restore
