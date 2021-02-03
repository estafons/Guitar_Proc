# from auxil import scan_folder_sep, eliminate_abrupt_keypoint_shifts, eliminate_low_confidence_keypoints, interpolate_keypoints, midhip_normalize, get_derivatives, centered_moving_average
from onsets.auxil import scan_folder_sep, eliminate_abrupt_keypoint_shifts, eliminate_low_confidence_keypoints, interpolate_keypoints, midhip_normalize, get_derivatives, centered_moving_average, str2bool
from skimage.transform import rescale
import numpy as np
import torch
from model import TCN
from torch.autograd import Variable
from . import visualize
# import visualize
import os
import librosa

import madmom
from madmom.features.onsets import peak_picking

import argparse


parser = argparse.ArgumentParser(description='AV Onset Detection')

parser.add_argument('--ksize', type=int, default=3,
					help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
					help='# of levels (default: 4)')
parser.add_argument('--nhid', type=int, default=150,
					help='number of hidden units per layer (default: 150)')
parser.add_argument('--dilations', type=str2bool, default=True,
					help='Choose weather to use simple cnn or with dilations')					

parser.add_argument('--fs', default=48000, type=int, action='store',
					help='Global audio sampling rate to use')
parser.add_argument('--hop', default=512, type=int, action='store',
					help='hop length')
parser.add_argument('--w_size', default=2048, type=int, action='store',
					help='window size')
parser.add_argument('--dropout', type=float, default=0.25,
					help='dropout applied to layers (default: 0.25)')					
args = parser.parse_args()


def run_audio(path_to_wav='./uploads/audio.wav'):
	audio_data_mix, sr = librosa.load(path_to_wav, sr=args.fs)
	print('Feature Exctraction')
	audio_feats = librosa.feature.melspectrogram(audio_data_mix, sr=args.fs, n_mels=40, n_fft=args.w_size, hop_length=args.hop)
	audio_feats = torch.Tensor(audio_feats.astype(np.float64))
	n_audio_channels = [args.nhid] * args.levels # e.g. [150] * 4

	print('Model Running...')
	model = TCN(40, 2, n_audio_channels, args.ksize, dropout=args.dropout, dilations=args.dilations)
	output = run_final_test(audio_feats, 'Audio', model)
	print('Model Reults ready!!')
	output = output.squeeze(0).cpu().detach()	
	print('output', output.size())
	oframes = peak_picking(activations=output[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
	otimes = librosa.core.frames_to_time(oframes, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)

	return otimes


def run_final_test(feats, input_type, model=None):
	model_name = "./static/TCN_"+input_type+"_0.pt"

	model = torch.load(model_name)

	with torch.no_grad():
		x = Variable(feats, requires_grad=True) # _greg_
		x = x.cuda()
		output = model(x.unsqueeze(0))
	
	return output

if __name__ == "__main__":
	run()