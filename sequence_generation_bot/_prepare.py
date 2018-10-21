from dataset import Dataset
from params.lermontov_azimov_vedmak_2 import ps
from params.azimov import ps as ps_1
from params.lermontov import ps as ps_2
from params.vedmak import ps as ps_3
dp = Dataset(ps)
dp.prepareData([ps_1, ps_2, ps_3])
pass