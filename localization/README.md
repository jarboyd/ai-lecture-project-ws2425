# AI lecture project WS24/25

This repository contains the source for the project submission for the lecture "Grundlagen der künstlichen Intelligenz WS24/25" at the [Technische Universität Clausthal](https://www.tu-clausthal.de/).

## Setup

The [GTSDB (German Traffic Sign Detection Benchmark) dataset](https://doi.org/10.17894/ucph.358970eb-0474-4d8f-90b5-3f124d9f9bc6) used in this project is not included in the source code and has to be manually downloaded from their website or using the `download.py` script.

After that, the `prepare.py` script should be used to create splits for training and testing.

The `train.py` script expects the full dataset to be in a folder named `FullIJCNN2013` containing images following the naming schema `00123.ppm` and a file `gt.txt` containing the ground truths for all images.

Adjust `hyperparams.py` to adjust some parameters across the whole process.
