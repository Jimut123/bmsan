import numpy as np, numpy.random
import drrmsan_multilosses

# Generating Cases

for i in range(3):
    alphas = np.random.dirichlet(np.ones(4),size=1)
    alpha_1 = float("%0.2f" % (alphas[0][0]))
    alpha_2 = float("%0.2f" % (alphas[0][1]))
    alpha_3 = float("%0.2f" % (alphas[0][2]))
    alpha_4 = float("%0.2f" % (alphas[0][3]))

    print(alpha_1," ",alpha_2," ",alpha_3," ",alpha_4," = ",alpha_1+alpha_2+alpha_3+alpha_4)
    dice = drrmsan_multilosses.get_dice_from_alphas(
        float(alpha_1), float(alpha_2), float(alpha_3), float(alpha_4))






    








