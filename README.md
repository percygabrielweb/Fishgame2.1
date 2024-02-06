# Fishgame2
This is a simulation for fish getting fed in a fish-farm. It works as a generator of data for training object detection AI systems.


# game.py:
Run this game to generate images and corresponding labels

# split_dataset.py
Run this after images and labels have been generated to create training, validation and test set

# make_movie.py
run this to create a .mp4 movie from the images in the 'dataset' NOTE: don't do this right after split_dataset, but rather after you have ran game.py