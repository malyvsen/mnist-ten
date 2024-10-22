# MNIST with 10 labeled examples
Proof that I can do machine learning 🌈


## The task
Use only one labeled example for each digit from the MNIST training set - the rest can be used unlabeled. Make a working classifier!


## To check it out
Without installing anything, you can take a look at the [data test](https://nbviewer.jupyter.org/github/malyvsen/mnist-ten/blob/master/data_test.ipynb), [validation](https://nbviewer.jupyter.org/github/malyvsen/mnist-ten/blob/master/validate.ipynb), [training](https://nbviewer.jupyter.org/github/malyvsen/mnist-ten/blob/master/train.ipynb), and [baseline implementation](baseline.ipynb "Best viewed locally - nbviewer is having difficulties") notebooks.

But if you do want to go deeper, read on!

### With Poetry
[Poetry](https://python-poetry.org) makes things easier, but it's not strictly necessary.
```sh
poetry install # install dependencies in a new venv
poetry shell # spawn a new shell within the venv
```

### Without Poetry
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ipython kernel install --name "mnist-ten" --user
```

### Example code
```python
import torch
from mnist_ten.data import test_loader
from mnist_ten.models import classifier, weights_path

classifier.load_state_dict(torch.load(weights_path))
classifier(next(iter(test_loader))) # classify one batch
```


## What I did
I used the unlabeled data to train a classifier on an auxiliary task: predicting how an image has been rotated/flipped. To be successful at this, the classifier needs to figure out what the different digits look like, and form a sensible early vision pipeline.

I then took the first few of that classifier's layers and trained a few new layers on top of them, this time to solve the main task - classify digits. The insights from the auxiliary task were re-used to help solve the main one, yielding an accuracy of 52%.

I also checked to see how well a simple nearest-neighbor classifier performs - indeed not bad! It also got an accuracy of 52%.

### The time it took
* 2 hours - basic data loading & baseline implementation
* 5 hours - programming model architecture, training & validation
* 2 hours - experimenting with data augmentation
* 3 hours - training models (without interaction)


## What didn't work
* Training the entire model, instead of just the head, on the main task
* Alternating between training on the main and auxiliary tasks
* A few different architectures & hyperparameters
* Data augmentation - it's there, but doesn't seem to be helping much


## Possible improvements
* Try many different architectures (especially simpler ones! I probably over-engineered the model)
* Try a mix between training the entire model and training just the head
* Experiment with learning rate scheduling and optimizers
* Use the label-smoothing technique from [this paper](https://arxiv.org/pdf/1904.12848.pdf)