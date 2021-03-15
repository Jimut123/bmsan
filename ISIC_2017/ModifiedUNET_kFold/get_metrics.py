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

max_jaccard = max(jacard); min_jaccard = min(jacard)
mean_jaccard = sum(jacard)/5.0
max_jaccard_diff = max_jaccard - mean_jaccard 
min_jaccard_diff = mean_jaccard - min_jaccard 

max_dice = max(dice); min_dice = min(dice)
mean_dice = sum(dice)/5.0
max_dice_diff = max_dice - mean_dice 
min_dice_diff = mean_dice - min_dice 

max_prec = max(prec); min_prec = min(prec)
mean_prec = sum(prec)/5.0
max_prec_diff = max_prec - mean_prec 
min_prec_diff = mean_prec - min_prec 

print("Min jaccard = {} Max jaccard = {}".format(min_jaccard,max_jaccard))
print("Min dice = {} Max dice = {}".format(min_dice,max_dice))
print("Min prec = {} Max prec = {}".format(min_prec,max_prec))
print("Mean Jacard = {} Max Jaccard Diff = {} Min Jaccard Diff = {} ".format(mean_jaccard, max_jaccard_diff, min_jaccard_diff))
print("Mean Dice = {} Max Dice Diff = {} Min Dice Diff = {} ".format(mean_dice, max_dice_diff, min_dice_diff))
print("Mean Precision = {} Max Precision Diff = {} Min Precision Diff = {} ".format(mean_prec, max_prec_diff, min_prec_diff))
print()
print()
print("Dice = {} +/- {} ".format(mean_dice, max(max_dice_diff, min_dice_diff)))
print("Jacard = {} +/- {} ".format(mean_jaccard, max(max_jaccard_diff, min_jaccard_diff)))
print("Precision = {} +/- {} ".format(mean_prec, max(max_prec_diff, min_prec_diff)))
