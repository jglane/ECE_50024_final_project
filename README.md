<img src="results/apple2orange_0/train.gif" style="width:25%">
<img src="results/dog2abstract_art_0/train.gif" style="width:25%">
<img src="results/tiger2lion_0/train.gif" style="width:25%">

# Getting Started

## Train a model
To train a model, run the following command:
```bash
$ /path/to/python train.py DATASET_NAME
```
where `DATASET_NAME` is the name of the dataset you want train on. The dataset could be one of the following cycleGAN datasets built into tensorflow:
- apple2orange
- summer2winter_yosemite
- horse2zebra
- monet2photo
- cezanne2photo
- ukiyoe2photo
- vangogh2photo
- maps
- facades
- iphone2dslr_flower

If you want to train on a custom dataset, you can create one using the `build_dataset.py` script. See the section below for more information.

When `train.py` is run, a new folder will be created in `results` containing the trained model and the training history.

## Create a new dataset
Suppose we want to create a new dataset containing images of tigers and lions. We can do this by running the following command:
```bash
$ /path/to/python build_dataset.py tiger lion
```
This command will create a new dataset in the `data` folder called `tiger2lion`. The dataset will contain roughly 1000 images of tigers and 1000 images of lions. The images are scraped from flickr using the 'tiger' and 'lion' keywords.

Next, we need to specify images for intermediate testing. Find the newly created folder in `tests` and add an image of the first class called `A.jpg` and an image of the second class called `B.jpg`. In this case, we would add an image of a tiger called `A.jpg` and an image of a lion called `B.jpg`. If you want to select images from the testing set, run the following command:
```bash
$ /path/to/python select_test_images.py DATASET_NAME
```
which will save 10 examples of each class from the testing set to the `tests` folder. Choose the best images from these examples and rename them to `A.jpg` and `B.jpg`.

Now, we can train a model on this dataset by running the following command:
```bash
$ /path/to/python train.py tiger2lion
```

## Test a model
To test a model, run the following command:
```bash
$ /path/to/python test.py MODEL_DIR
```
where `MODEL_DIR` is the path to the model directory, usually within the `results` directory.

## Notes DELETE LATER
Tried a couple of times without identity loss but it was worse