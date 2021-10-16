# COVID-19 Detection in lung CT-scan images

The purpose of this project is to detect COVID-19 positive cases based on lung Ct scans images. The trained
model is based on a Resnext101 architecture and is fine-tuned in order to classify
 CT scan images.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages

```bash
pip3 install -r requirements.txt
```
If you're not placed on root folder, use full/relative path to requirements.txt instead.

## Data
Before running the program, you need to download the data first. You can find
it [here](https://www.kaggle.com/luisblanche/covidct).

Once you download the data, place the ``/train`` and ``/test`` folders in ``/data/covid_data``  folder
, Otherwise you will need to change ``train_path`` and ``test_path`` arguments when running the program. You should have
 a similar structure: 
```
Covid-19-Detection-CTScans
│   README.md
│   requirements.txt    
│
└───data
│   │   __init__.py
│   │   data.py
│   │
│   └───covid_data
│       │
│       │───train
│       │   └───img.png ....
│       │
│       └───test
│           └───img.png ....      
└───....
```

## Usage



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
