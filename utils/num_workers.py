from time import time
import multiprocessing as mp
from torch.utils.data import DataLoader
from utils.datasets import Remote14

train_dataset = Remote14(root_dir='data/remote14', is_train=True, descriptions_file="descriptions/train_descriptions_concise.json")
print(mp.cpu_count())
for num_workers in range(2, mp.cpu_count(), 2):
    train_loader = DataLoader(train_dataset,num_workers=num_workers,batch_size=4,pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))