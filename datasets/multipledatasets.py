import sys
import os

from main import setup
from options.cfg_options import CFGOptions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from termcolor import cprint
from datasets.comphand import CompHand
from datasets.freihand import FreiHAND
from datasets.ge import Ge  # Add Ge dataset import
from build import DATA_REGISTRY


@DATA_REGISTRY.register()
class MultipleDatasets(Dataset):
    def __init__(self, cfg, phase='train', writer=None):
        self.cfg = cfg
        self.dbs = []
        if self.cfg.DATA.FREIHAND.USE:
            self.dbs.append(FreiHAND(self.cfg, phase, writer))
        if self.cfg.DATA.COMPHAND.USE:
            self.dbs.append(CompHand(self.cfg, phase, writer))
        if hasattr(self.cfg.DATA, 'GE') and self.cfg.DATA.GE.USE:  # Add Ge dataset
            self.dbs.append(Ge(self.cfg, phase, writer))

        # Verify that all datasets implement the __getitem__ method
        for i, db in enumerate(self.dbs):
            if not hasattr(db, '__getitem__') or not callable(getattr(db, '__getitem__')):
                raise NotImplementedError(
                    f"Dataset at index {i} ({db.__class__.__name__}) does not implement __getitem__ method"
                )

        self.db_num = len(self.dbs)
        if self.db_num == 0:
            raise ValueError("No datasets were loaded. Please check your configuration.")

        self.max_db_data_num = max([len(db) for db in self.dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])
        self.make_same_len = False
        if writer is not None:
            writer.print_str('Merge train set, total {} samples'.format(self.__len__()))
        cprint('Merge train set, total {} samples'.format(self.__len__()), 'red')

    def __len__(self):
        # All datasets have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # Each dataset has a different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])):  # last batch: random sampling
                data_idx = random.randint(0, len(self.dbs[db_idx]) - 1)
            else:  # before the last batch: use modulo
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            # Add boundary check
            if index >= self.__len__():
                raise IndexError(f"Index {index} out of range for dataset of length {self.__len__()}")

            db_idx = 0
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx - 1]

        # Add error handling
        try:
            return self.dbs[db_idx][data_idx]
        except NotImplementedError:
            raise NotImplementedError(
                f"Dataset {self.dbs[db_idx].__class__.__name__} does not implement __getitem__ properly"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error accessing item {data_idx} from dataset {self.dbs[db_idx].__class__.__name__}: {str(e)}"
            )

