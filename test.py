'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import *
from utils import load_datasets, make_dataloaders
import time
import gc
import json

_, _, test_set = load_datasets(c.dataset_path, c.class_name, test=True)
_, _, test_loader = make_dataloaders(None, None, test_set, test=True)

model = torch.load("models/" + c.modelname + "", map_location=torch.device('cpu'))

with open('models/' + c.modelname + '.json') as jsonfile:
    model_parameters = json.load(jsonfile)

time_start = time.time()
test(model, model_parameters, test_loader)
time_end = time.time()
time_c = time_end - time_start  # 运行所花时间

print("test time cost: {:f} s".format(time_c))