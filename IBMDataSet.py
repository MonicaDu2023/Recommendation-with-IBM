#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch.utils.data

class IBMDataset(torch.utils.data.Dataset):
    """
        IBM Dataset
        Data preparation: treat samples with a rating less than 3 as negative samples
    """

    def __init__(self, dataset):
        # Read the data into a Pandas dataframe
        data = dataset.to_numpy()[:, :3]

        # Retrieve the items and ratings data
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)

        # Get the range of the items
        self.field_dims = np.max(self.items, axis=0) + 1

        # Initialize NumPy arrays to store user and item indices
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        """
        :return: number of total ratings
        """
        return self.targets.shape[0]

    def __getitem__(self, index):
        """
        :param index: current index
        :return: the items and ratings at current index
        """
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        """
        :param target: ratings
        :return: binary ratings (0 or 1)
        """
        return target

