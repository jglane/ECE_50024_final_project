# Getting Started

![](results/summer2winter_yosemite_0/train.gif)

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
- cityscapes
- facades

If you want to train on a custom dataset, you can create one using the `scrape_images.py` script. See the section below for more information.

When `train.py` is run, new folder will be created in `results` containing the trained model and the training history.

## Create a new dataset
Suppose we want to create a new dataset containing images of tigers and lions. We can do this by running the following command:
```bash
$ /path/to/python scrape_images.py tiger lion
```
This command will create a new dataset in the `data` folder called `tiger2lion`. The dataset will contain roughly 1000 images of tigers and 1000 images of lions. The images are scraped from flickr using the 'tiger' and 'lion' keywords.

Now, we can train a model on this dataset by running the following command:
```bash
$ /path/to/python train.py tiger2lion
```