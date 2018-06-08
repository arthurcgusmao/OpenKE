import os
from tools import generate_g_hat

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

generate_g_hat.run(import_path='./results/WN11/TransE/1527008520', ks=[3,5,7])
