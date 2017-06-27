
## Installation
```bash
git clone https://github.com/beeva-samuelmunoz/hci-eye_tracking-models.git
cd hci-eye_tracking-models
mkdir data
virtualenv -p python3 venv3
source venv3/bin/activate
pip install -r requirements.txt
```

## Dataset
1. Download the  [features01_dlib dataset](https://drive.google.com/open?id=0B4BwXne65MbQVld4clV4SUlWdEk) into the `data` folder.
1. Decompress the file `7z x data/HCI-ET-dataset_features01_dlib_augmented-v02.7z`


# TODO
1. Balance x,y targets (Ric)
