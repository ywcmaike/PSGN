import torch
import os
import time

class Saver:
    def __init__(self, save_path):
        self.save_path = save_path
        self.save_files = []
    
    def save_optimizer(self, optimizer_dict, n_epoch):
        optimizer_file = 'optimizer_epoch' + str(n_epoch) + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())  + '.pth'
        optimizer_file = os.path.join(self.save_path, optimizer_file)
        torch.save(optimizer_dict, optimizer_file)

    def save_model(self, model_dict, n_epoch):
        model_file = 'model_epoch' + str(n_epoch) + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())  + '.pth'
        model_file = os.path.join(self.save_path, model_file)
        self.save_files.append(model_file)
        torch.save(model_dict, model_file)

def create_saver(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return Saver(save_path)