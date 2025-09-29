import torch
import numpy as np
from sklearn import metrics
from operator import itemgetter


def save_parameters(self, path):
		torch.save(self.state_dict(), path)

def load_parameters(self, path):
    self_state = self.state_dict()
    loaded_state = torch.load(path)
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "").replace("speaker_encoder.", "")
            if name not in self_state:
                print("%s is not in the model."%origname)
                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)



'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F

def init_args(args):
	args.score_save_path    = os.path.join(args.save_path, 'score.txt')
	args.model_save_path    = os.path.join(args.save_path, 'model')
	os.makedirs(args.model_save_path, exist_ok = True)
	return args

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = [];
	if target_fr:
		for tfr in target_fr:
			idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	
	return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res




def evaluate(model, loader, device, p_target=0.01, c_miss=1, c_fa=1):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for audios, labels in loader:
            audios = audios.to(device)
            labels = labels.to(device)

            embeds = model(audios)  # (B, D)
            embeds = torch.nn.functional.normalize(embeds, dim=1)

            # Cosine similarity matrix
            sim = embeds @ embeds.t()

            # Collect positive/negative pairs
            B = labels.size(0)
            for i in range(B):
                for j in range(i+1, B):
                    score = sim[i, j].item()
                    all_scores.append(score)
                    all_labels.append(1 if labels[i] == labels[j] else 0)

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # Compute FPR, FNR, thresholds
    tunedThreshold, eer, fpr, fnr = tuneThresholdfromScore(scores, labels, target_fa=[0.01, 0.05, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    min_dcf, min_dcf_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    return {
        "EER": eer,
        "MinDCF": min_dcf,
        "Thresholds": tunedThreshold
    }
