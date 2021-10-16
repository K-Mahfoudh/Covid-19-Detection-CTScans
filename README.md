# COVID-19 Detection in lung CT-scan images

The purpose of this project is to detect COVID-19 positive cases based on lung CT scan images. The 
model is based on a Resnext101 architecture and is fine-tuned in order to classify our own images.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages

```bash
pip3 install -r requirements.txt
```
If you're not placed in the root folder, use full/relative path to requirements.txt instead.

**Important**: Make sure to install the required packages, otherwise, you may 
have some package related errors (especially CUDA issues) while running the program.

## Data
Before running the program, you need to download the data first. You can find
it [here](https://www.kaggle.com/luisblanche/covidct).

Once you download the data, place the ``/train`` and ``/test`` folders in ``/data/covid_data``  folder
, Otherwise you will need to change ``train_path`` and ``test_path`` arguments when running your code. You should have
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
You can run the program using the following command:

```
python3 covid19_detector.py
```
This command will use the default parameters, if you want to change any parameter
, you will need to pass it as an argument when running the program, for example:
 ```
 python3 covid19_detector.py -m Test -lr 0.001
 ```
For more information about arguments, type the following command:
```
python3 covid19_detector.py -h
```
## Errors and Logging

Thrown errors and exceptions during training/ testing are written to a log file ``/logs/error.log``.
Make sure to check the logs if you have any issues.
## Contributing
All are welcome to contribute, but before making any changes, please make sure
to check the project's style. It is best if you follow the "fork-and-pull" Git workflow.

1. Fork the repo on GitHub
2. Clone the project to your own machine
3. Commit changes to your own branch
4. Push your work back up to your fork
5. Submit a Pull request.

**Important**: Do not forget to keep your cloned project up to date, and to open
a new issue in case of any major changes.

## License
[MIT](https://github.com/K-Mahfoudh/Covid-19-Detection-CTScans/blob/main/LICENSE.md)
