# Monodeep
In development...

# Kitti Dataset
Most Indicated scenes list (without static scenes):

**City**: All except 0017, 0018, 0057, 0060, 0002, 0026

**Residential**: All

**Road**: All

**Campus**: Only 0038, 0045, 0047

**Person**: N/a

# Training
Ex: 
    
    ./monodeep.py -m train -s kitti2012 --max_steps 1000 -t -d 0.5
    ./monodeep.py -m train -s kitti2015 --max_steps 1000 -t -d 0.5
    ./monodeep.py -m train -s kittiraw_campus --max_steps 1000 -t -d 0.5
    ./monodeep.py -m train -s nyudepth --max_steps 100 -d 0.5 --ldecay
    ./monodeep.py -m train -s kittiraw_residential_continuous --max_steps 10 -d 0.5 --ldecay -t --gpu 0

# Testing/Restore
Ex: 

    ./monodeep.py -m test -s kitti2012 -r output/monodeep/2018-01-27_14-20-07/restore/
    ./monodeep.py -m test -s kittiraw_residential_continuous -r output/monodeep/2018-02-09_15-21-56/restore/ -u 
    
# Dataset Prepation Script

It's no longer necessary to generate dataset.pkl for training. The `monodeep_dataloader.py` identifies the images available for training and testing on-the-fly.

Ex: 

    python3 dataset_preparation.py -s kittiRaw  (Deprecated)
    python3 dataset_preparation.py -s kitti2012 (Deprecated)
    python3 dataset_preparation.py -s kitti2015 (Deprecated)
    python3 dataset_preparation.py -s nyuDepth  (Deprecated)
