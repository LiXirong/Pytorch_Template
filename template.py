# -*- coding: utf-8 -* -

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
#============================#
from networks import MyModel
#============================#

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,1]

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    #============================#
    model = MyModel()
    #============================#
    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    criterion = xxxx
    criterion.to(device)

    optimizer = optim.xxxx

    def _worker_init_fn_():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32-1
        random.seed(torch_seed)
        np.random.seed(np_seed)
    transform = transforms.Compose([xxxxxx])
    dataset = xxxxxx
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=xxxx,
        shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_())    

    num_epochs = xxxx
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        for i, data in enumerate(dataloader, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            image, label = data
            image, label = image.to(device), label.to(device)

            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            if i % 10 == 9:
                model.eval()
                with torch.no_grad():
                    #####some testing#####
                    print("xxxxxxx".format(xxxxxxx))
        # the end of one epoch
        model.eval()
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, xxPATHxx)
        with torch.no_grad():
            #####some testing#####
            rint("xxxxxxx".format(xxxxxxx))
            #####some logging#####
