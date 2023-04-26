# Getting Started

## Train a model
To train a model, run the following command:
```bash
/path/to/python train.py
```

## Create a new dataset
Suppose we want to create a new dataset containing images of tigers and lions. We can do this by running the following command:
```bash
/path/to/python scrape_images.py tiger lion
```
This command will create a new dataset in the `data` folder called `tiger2lion`. The dataset will contain roughly 1000 images of tigers and 1000 images of lions. The images are scraped from flickr using the 'tiger' and 'lion' keywords.