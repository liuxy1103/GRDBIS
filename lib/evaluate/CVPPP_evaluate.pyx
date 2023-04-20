from __future__ import division
from libcpp cimport bool as bool_t
import cv2
import math
import numpy as np
cimport numpy as np
cimport cython

ctypedef bint TYPE_BOOL
ctypedef unsigned long long TYPE_U_INT64
ctypedef unsigned int TYPE_U_INT32
ctypedef unsigned short TYPE_U_INT16
ctypedef unsigned char TYPE_U_INT8
ctypedef long long TYPE_INT64
#ctypedef long TYPE_INT32
ctypedef int TYPE_INT32
ctypedef short TYPE_INT16
ctypedef signed char TYPE_INT8
ctypedef float TYPE_FLOAT
ctypedef double TYPE_DOUBLE


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def DiffFGLabels(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: difference of the number of foreground labels

    # check if label images have same size
    # if (inLabel.shape != gtLabel.shape):
    #     return -1

    cdef int maxInLabel = np.int(np.max(inLabel)) # maximum label value in inLabel
    cdef int minInLabel = np.int(np.min(inLabel)) # minimum label value in inLabel
    cdef int maxGtLabel = np.int(np.max(gtLabel)) # maximum label value in gtLabel
    cdef int minGtLabel = np.int(np.min(gtLabel)) # minimum label value in gtLabel

    cdef double out = (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def BestDice(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel):
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

    cdef int i, j
    cdef double sMax = 0.0
    cdef double s = 0.0
    cdef double score = 0.0 # initialize output

    # check if label images have same size
    # if (inLabel.shape != gtLabel.shape):
    #     return score

    cdef int maxInLabel = np.max(inLabel) # maximum label value in inLabel
    cdef int minInLabel = np.min(inLabel) # minimum label value in inLabel
    cdef int maxGtLabel = np.max(gtLabel) # maximum label value in gtLabel
    cdef int minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

    if(maxInLabel == minInLabel): # trivial solution
        return score

    for i in range(minInLabel+1, maxInLabel+1): # loop all labels of inLabel, but background
        sMax = 0; # maximum Dice value found for label i so far
        for j in range(minGtLabel+1, maxGtLabel+1): # loop all labels of gtLabel, but background
            s = Dice(inLabel, gtLabel, i, j) # compare labelled regions
            # keep max Dice value for label i
            if(sMax < s):
                sMax = s
        score = score + sMax; # sum up best found values
    score = score / (maxInLabel-minInLabel)
    return score


@cython.boundscheck(False)
@cython.wraparound(False)
def FGBGDice(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel):
# input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
#        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
# output: Dice score for foreground/background segmentation, only.

    # check if label images have same size
    # if (inLabel.shape != gtLabel.shape):
    #     return 0

    cdef int minInLabel = np.min(inLabel) # minimum label value in inLabel
    cdef int minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

    cdef np.ndarray[TYPE_U_INT16, ndim=2] one = np.ones_like(inLabel)
    cdef np.ndarray[TYPE_U_INT16, ndim=2] inFgLabel = (inLabel != minInLabel*one)*one
    cdef np.ndarray[TYPE_U_INT16, ndim=2] gtFgLabel = (gtLabel != minGtLabel*one)*one

    cdef double out = Dice(inFgLabel,gtFgLabel,1,1) # Dice score for the foreground
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def Dice(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel, int i, int j):
# calculate Dice score for the given labels i and j
    
    # check if label images have same size
    # if (inLabel.shape != gtLabel.shape):
    #     return 0

    cdef double out = 0.0
    cdef np.ndarray[TYPE_U_INT16, ndim=2] one = np.ones_like(inLabel)
    # cdef np.ndarray[TYPE_U_INT16, ndim=2] inMask = (inLabel==i*one) # find region of label i in inLabel
    # cdef np.ndarray[TYPE_U_INT16, ndim=2] gtMask = (gtLabel==j*one) # find region of label j in gtLabel
    cdef int inSize = np.sum((inLabel==i*one)*one) # cardinality of set i in inLabel
    cdef int gtSize = np.sum((gtLabel==j*one)*one) # cardinality of set j in gtLabel
    cdef int overlap= np.sum((inLabel==i*one)*(gtLabel==j*one)*one) # cardinality of overlap of the two regions
    if ((inSize + gtSize)>1e-8):
        out = 2*overlap/(inSize + gtSize) # Dice score
    else:
        out = 0
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def AbsDiffFGLabels(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: Absolute value of difference of the number of foreground labels

    cdef double out = np.abs(DiffFGLabels(inLabel, gtLabel))
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def SymmetricBestDice(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel):
# SBD(inLabel, gtLabel) = min{BD(inLabel, gtLabel), BD(gtLabel, inLabel)}

    cdef double bd1 = BestDice(inLabel,gtLabel)
    cdef double bd2 = BestDice(gtLabel,inLabel)
    if bd1 < bd2:
        return bd1
    else:
        return bd2

@cython.boundscheck(False)
@cython.wraparound(False)
def SymmetricBestDice_max(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel):
# SBD(inLabel, gtLabel) = min{BD(inLabel, gtLabel), BD(gtLabel, inLabel)}

    cdef double bd1 = BestDice(inLabel,gtLabel)
    cdef double bd2 = BestDice(gtLabel,inLabel)
    if bd1 > bd2:
        return bd1
    else:
        return bd2

@cython.boundscheck(False)
@cython.wraparound(False)
def SymmetricBestDice_both(np.ndarray[TYPE_U_INT16, ndim=2] inLabel, np.ndarray[TYPE_U_INT16, ndim=2] gtLabel):
# SBD(inLabel, gtLabel) = min{BD(inLabel, gtLabel), BD(gtLabel, inLabel)}

    cdef double bd1 = BestDice(inLabel,gtLabel)
    cdef double bd2 = BestDice(gtLabel,inLabel)
    if bd1 > bd2:
        sbd_min = bd2
        sbd_max = bd1
    else:
        sbd_min = bd1
        sbd_max = bd2
    return sbd_min, sbd_max
