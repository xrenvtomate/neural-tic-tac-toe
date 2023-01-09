# neural tic tac toe

## setup
you need python3 with jupyter notebook installed (3.10 recommended) 
```bash
git clone https://github.com/xrenvtomate/neural-tic-tac-toe
cd neural-tic-tac-toe
```

### create and activate venv

#### linux/macos
```bash
python3 -m venv venv
source venv/bin/activate
```
#### windows
```bash
python -m venv venv
python venv\scripts\activate
```
### install dependencies
```bash
pip install -r requirements.txt
```

## train
set hyperparameters in ./nnconfig.py
then use train.ipynb or 
```bash
python train.py
```
to train  and view test statisctics

## play
```bash
python main.py 
```
to play with last saved parameters
or
```bash
python main.py [dump folder]
```
there is already two dumps in the repo


## neural net architecture
### input layer
18 input neurons (is there x in cell 1, ..., is there x in cell 9, is there o in cell 1, ..., is tehre o in cell 9)
### one hidden layer
hidden layer with number of neurons defined in nnconfig.py and sigmoid as activation function
### output layer
9 neurons with sigmoid and activation function
(profit of move to first cell, ... profit of move to nineth cell)