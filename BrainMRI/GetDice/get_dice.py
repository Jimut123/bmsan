import numpy as np, numpy.random
import drrmsan_multilosses
import pickle
import pickle5 as pickle

# Generating Cases


with open('alpha_datas.pickle', 'rb') as handle:
    data = pickle.load(handle)

print(data)

count = 1

for item in data:
    if count <= 5:
        count += 1
        continue
    alpha_1 = float("%0.2f" % (item[0][0]))
    alpha_2 = float("%0.2f" % (item[0][1]))
    alpha_3 = float("%0.2f" % (item[0][2]))
    alpha_4 = float("%0.2f" % (item[0][3]))
    print(alpha_1," ",alpha_2," ",alpha_3," ",alpha_4," = ",alpha_1+alpha_2+alpha_3+alpha_4)
    dice = drrmsan_multilosses.get_dice_from_alphas(
        float(alpha_1), float(alpha_2), float(alpha_3), float(alpha_4))

