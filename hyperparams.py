# preprocessing
# convert images from .ppm to .jpg; only needs to be done once
convert = True
# shuffle images before splitting
shuffle_files = True
# use partial dataset for training; must be in range (0,1)
train_split = 0.8

# training
# upsample images, heavy RAM usage; must be an integer
upsample_limit = 2

# performance
# use only partial dataset to reduce RAM usage; must be in range (0,1)
perf_take = 1
