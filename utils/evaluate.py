import numpy as np

##############################################################################
def DiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return -1

    maxInLabel = np.int(np.max(inLabel)) # maximum label value in inLabel
    minInLabel = np.int(np.min(inLabel)) # minimum label value in inLabel
    maxGtLabel = np.int(np.max(gtLabel)) # maximum label value in gtLabel
    minGtLabel = np.int(np.min(gtLabel)) # minimum label value in gtLabel

    return  (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel) 

##############################################################################
def BestDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: score: Dice score
#
# We assume that the lowest label in inLabel is background, same for gtLabel
# and do not use it. This is necessary to avoid that the trivial solution, 
# i.e. finding only background, gives excellent results.
#
# For the original Dice score, labels corresponding to each other need to
# be known in advance. Here we simply take the best matching label from 
# gtLabel in each comparison. We do not make sure that a label from gtLabel
# is used only once. Better measures may exist. Please enlighten me if I do
# something stupid here...

    score = 0 # initialize output
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return score
    
    maxInLabel = np.max(inLabel) # maximum label value in inLabel
    minInLabel = np.min(inLabel) # minimum label value in inLabel
    maxGtLabel = np.max(gtLabel) # maximum label value in gtLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel
    
    if(maxInLabel==minInLabel): # trivial solution
        return score
    
    for i in range(minInLabel+1,maxInLabel+1): # loop all labels of inLabel, but background
        sMax = 0; # maximum Dice value found for label i so far
        for j in range(minGtLabel+1,maxGtLabel+1): # loop all labels of gtLabel, but background
            s = Dice(inLabel, gtLabel, i, j) # compare labelled regions
            # keep max Dice value for label i
            if(sMax < s):
                sMax = s
        score = score + sMax; # sum up best found values
    score = score/(maxInLabel-minInLabel)
    return score

##############################################################################
def FGBGDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
#        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
# output: Dice score for foreground/background segmentation, only.

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    minInLabel = np.min(inLabel) # minimum label value in inLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

    one = np.ones(inLabel.shape)    
    inFgLabel = (inLabel != minInLabel*one)*one
    gtFgLabel = (gtLabel != minGtLabel*one)*one
    
    return Dice(inFgLabel,gtFgLabel,1,1) # Dice score for the foreground

##############################################################################
def Dice(inLabel, gtLabel, i, j):
# calculate Dice score for the given labels i and j
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel==i*one) # find region of label i in inLabel
    gtMask = (gtLabel==j*one) # find region of label j in gtLabel
    inSize = np.sum(inMask*one) # cardinality of set i in inLabel
    gtSize = np.sum(gtMask*one) # cardinality of set j in gtLabel
    overlap= np.sum(inMask*gtMask*one) # cardinality of overlap of the two regions
    if ((inSize + gtSize)>1e-8):
        out = 2*overlap/(inSize + gtSize) # Dice score
    else:
        out = 0

    return out

##############################################################################
def AbsDiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: Absolute value of difference of the number of foreground labels

    return np.abs( DiffFGLabels(inLabel,gtLabel) )

##############################################################################
def SymmetricBestDice(inLabel,gtLabel):
# SBD(inLabel, gtLabel) = min{BD(inLabel, gtLabel), BD(gtLabel, inLabel)}

    bd1 = BestDice(inLabel,gtLabel)
    bd2 = BestDice(gtLabel,inLabel)
    if bd1 < bd2:
        return bd1
    else:
        return bd2
