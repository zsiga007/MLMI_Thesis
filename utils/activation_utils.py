import torch
import numpy as np
import random
import transformers
from tqdm import tqdm
import json
import wandb
import time
from torch.utils.data.distributed import DistributedSampler
import os
import re

# a function that takes in a llama 2 7b model, gets its activations from a list of layers, and returns the activations
def get_activations(model, layers):
    # create a dictionary to store the activations
    activations = {}
    # loop through the layers
    for layer in layers:
        
    # return the activationsa
    return activations