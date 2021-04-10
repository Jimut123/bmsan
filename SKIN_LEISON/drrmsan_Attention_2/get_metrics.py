import numpy as np
import math


with open('Output.txt') as f:
    lines = [line.rstrip() for line in f]
jacard = []
dice = []
prec = []
for item in lines:
    #print(item.split(' '))
    split_ = item.split(' ')
    jacard.append(float(split_[5]))
    dice.append(float(split_[9]))
    prec.append(float(split_[13]))
print(jacard)
print(dice)
print(prec)

N = 0
dice_sum = 0
for item in dice:
    if np.isnan(item) == False:
        dice_sum += item
        N += 1
mean_dice = float(dice_sum/N)
sum_sig = 0
for item in dice:
    if np.isnan(item) == False:
        sum_sig += (item - mean_dice)*(item - mean_dice)
sum_sig /= N
sd_dice = math.sqrt(sum_sig)



N = 0
jacard_sum = 0
for item in jacard:
    if np.isnan(item) == False:
        jacard_sum += item
        N += 1
mean_jacard = float(jacard_sum/N)
sum_sig_jac = 0
for item in jacard:
    if np.isnan(item) == False:
        sum_sig_jac += (item - mean_jacard)*(item - mean_jacard)
sum_sig_jac /= N
sd_jacard = math.sqrt(sum_sig_jac)



N = 0
prec_sum = 0
for item in prec:
    if np.isnan(item) == False:
        prec_sum += item
        N += 1
mean_prec = float(prec_sum/N)
sum_sig_prec = 0
for item in prec:
    if np.isnan(item) == False:
        sum_sig_prec += (item - mean_prec)*(item - mean_prec)
sum_sig_prec /= N
sd_precision = math.sqrt(sum_sig_prec)

print("Dice = {} +/- {} ".format(mean_dice*100, sd_dice*100))
print("Jacard = {} +/- {} ".format(mean_jacard*100, sd_jacard*100))
print("Precision = {} +/- {} ".format(mean_prec*100, sd_precision*100))
