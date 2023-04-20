#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
import h5py


def evaluate(output, target):
    diffFG = DiffFGLabels(output,target)
    bestDice = BestDice(output,target)
    return diffFG,bestDice




##############################################################################
def DiffFGLabels(inLabel, gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape != gtLabel.shape):
        return -1

    maxInLabel = np.int(np.max(inLabel))  # maximum label value in inLabel
    minInLabel = np.int(np.min(inLabel))  # minimum label value in inLabel
    maxGtLabel = np.int(np.max(gtLabel))  # maximum label value in gtLabel
    minGtLabel = np.int(np.min(gtLabel))  # minimum label value in gtLabel

    return (maxInLabel - minInLabel) - (maxGtLabel - minGtLabel)


##############################################################################
def BestDice(inLabel, gtLabel):
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

    score = 0  # initialize output

    # check if label images have same size
    if (inLabel.shape != gtLabel.shape):
        return score

    maxInLabel = np.max(inLabel)  # maximum label value in inLabel
    minInLabel = np.min(inLabel)  # minimum label value in inLabel
    maxGtLabel = np.max(gtLabel)  # maximum label value in gtLabel
    minGtLabel = np.min(gtLabel)  # minimum label value in gtLabel

    if (maxInLabel == minInLabel):  # trivial solution
        return score

    for i in range(minInLabel + 1, maxInLabel + 1):  # loop all labels of inLabel, but background
        sMax = 0;  # maximum Dice value found for label i so far
        for j in range(minGtLabel + 1, maxGtLabel + 1):  # loop all labels of gtLabel, but background
            s = Dice(inLabel, gtLabel, i, j)  # compare labelled regions
            # keep max Dice value for label i
            if (sMax < s):
                sMax = s
        score = score + sMax;  # sum up best found values
    score = score / (maxInLabel - minInLabel)
    return score


##############################################################################
def FGBGDice(inLabel, gtLabel):
    # input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
    #        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
    # output: Dice score for foreground/background segmentation, only.

    # check if label images have same size
    if (inLabel.shape != gtLabel.shape):
        return 0

    minInLabel = np.min(inLabel)  # minimum label value in inLabel
    minGtLabel = np.min(gtLabel)  # minimum label value in gtLabel

    one = np.ones(inLabel.shape)
    inFgLabel = (inLabel != minInLabel * one) * one
    gtFgLabel = (gtLabel != minGtLabel * one) * one

    return Dice(inFgLabel, gtFgLabel, 1, 1)  # Dice score for the foreground


##############################################################################
def Dice(inLabel, gtLabel, i, j):
    # calculate Dice score for the given labels i and j

    # check if label images have same size
    if (inLabel.shape != gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel == i * one)  # find region of label i in inLabel
    gtMask = (gtLabel == j * one)  # find region of label j in gtLabel
    inSize = np.sum(inMask * one)  # cardinality of set i in inLabel
    gtSize = np.sum(gtMask * one)  # cardinality of set j in gtLabel
    overlap = np.sum(inMask * gtMask * one)  # cardinality of overlap of the two regions
    if ((inSize + gtSize) > 1e-8):
        out = 2 * overlap / (inSize + gtSize)  # Dice score
    else:
        out = 0

    return out


##############################################################################
def AbsDiffFGLabels(inLabel, gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: Absolute value of difference of the number of foreground labels

    return np.abs(DiffFGLabels(inLabel, gtLabel))


##############################################################################
def WriteOutput(output_filename, somedict):
    # output_filename: name of the output file
    # results: array containing the result values
    output_file = open(output_filename, 'w')
    # write header if available
    if 'header' in somedict:
        output_file.write('dataset,')
        output_file.write(','.join(map(str, somedict['header'])))
        output_file.write('\n')
    # write rest
    for key in somedict:
        # skip header
        if key.find('header') == -1:
            # get every line and output it
            for line in somedict[key]:
                output_file.write(key)
                output_file.write(',')
                output_file.write(','.join(map(str, line)))
                output_file.write('\n')

    output_file.close()


##############################################################################
# main routine
if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print(submit_dir + " doesn't exist")

    # check if input directories exist
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        # create output dir if not already there
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # the input filenames
        truth_file = os.path.join(truth_dir, "CVPPP2017_testing_truth.h5")
        submission_answer_file = os.path.join(submit_dir, "answer.h5")

        # call the evaluation routine dealing with these files
        results, stats = evaluate(submission_answer_file, truth_file)

        # write results to output file
        output_filename = os.path.join(output_dir, 'details.txt')
        WriteOutput(output_filename, results)
        output_filename = os.path.join(output_dir, 'scores.txt')
        WriteOutput(output_filename, stats)
