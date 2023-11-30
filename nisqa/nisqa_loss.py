import numpy as np
import torch
import NISQA_lib as NL
import argparse
import glob
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

class NISQALoss(nn.Module):
		def __init__(self, device):
				super(NISQALoss,self).__init__()
				self.device = device
				args = {}
				args['pretrained_model'] = 'nisqa.tar'

				checkpoint = torch.load(args['pretrained_model'], map_location='cuda')
				checkpoint['args'].update(args)
				self.args = checkpoint['args']
				self.args['ms_sr'] = 16000
				self.args['ms_fmax'] = 8000
				self.args['ms_max_segments'] = 500

				model_args = {
								    
								    'ms_seg_length': self.args['ms_seg_length'],
								    'ms_n_mels': self.args['ms_n_mels'],
								    
								    'cnn_model': self.args['cnn_model'],
								    'cnn_c_out_1': self.args['cnn_c_out_1'],
								    'cnn_c_out_2': self.args['cnn_c_out_2'],
								    'cnn_c_out_3': self.args['cnn_c_out_3'],
								    'cnn_kernel_size': self.args['cnn_kernel_size'],
								    'cnn_dropout': self.args['cnn_dropout'],
								    'cnn_pool_1': self.args['cnn_pool_1'],
								    'cnn_pool_2': self.args['cnn_pool_2'],
								    'cnn_pool_3': self.args['cnn_pool_3'],
								    'cnn_fc_out_h': self.args['cnn_fc_out_h'],
								    
								    'td': self.args['td'],
								    'td_sa_d_model': self.args['td_sa_d_model'],
								    'td_sa_nhead': self.args['td_sa_nhead'],
								    'td_sa_pos_enc': self.args['td_sa_pos_enc'],
								    'td_sa_num_layers': self.args['td_sa_num_layers'],
								    'td_sa_h': self.args['td_sa_h'],
								    'td_sa_dropout': self.args['td_sa_dropout'],
								    'td_lstm_h': self.args['td_lstm_h'],
								    'td_lstm_num_layers': self.args['td_lstm_num_layers'],
								    'td_lstm_dropout': self.args['td_lstm_dropout'],
								    'td_lstm_bidirectional': self.args['td_lstm_bidirectional'],
								    
								    'td_2': self.args['td_2'],
								    'td_2_sa_d_model': self.args['td_2_sa_d_model'],
								    'td_2_sa_nhead': self.args['td_2_sa_nhead'],
								    'td_2_sa_pos_enc': self.args['td_2_sa_pos_enc'],
								    'td_2_sa_num_layers': self.args['td_2_sa_num_layers'],
								    'td_2_sa_h': self.args['td_2_sa_h'],
								    'td_2_sa_dropout': self.args['td_2_sa_dropout'],
								    'td_2_lstm_h': self.args['td_2_lstm_h'],
								    'td_2_lstm_num_layers': self.args['td_2_lstm_num_layers'],
								    'td_2_lstm_dropout': self.args['td_2_lstm_dropout'],
								    'td_2_lstm_bidirectional': self.args['td_2_lstm_bidirectional'],                
								    
								    'pool': self.args['pool'],
								    'pool_att_h': self.args['pool_att_h'],
								    'pool_att_dropout': self.args['pool_att_dropout'],
								    }

				self.model = NL.NISQA_DIM(**model_args)     
				missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
				print('Loaded pretrained model from ' + self.args['pretrained_model'])
				if missing_keys:
						print('missing_keys:')
						print(missing_keys)
				if unexpected_keys:
						print('unexpected_keys:')
						print(unexpected_keys)        

				self.model.to(self.device)
				self.model.eval()
				self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.args['ms_sr'], n_fft=self.args['ms_n_fft'], 
								                                         hop_length=int(self.args['ms_sr']*self.args['ms_hop_length']), 
								                                         win_length=int(self.args['ms_sr']*self.args['ms_win_length']), 
								                                         n_mels=self.args['ms_n_mels'], f_max=self.args['ms_fmax'], 
								                                         power=1.0,
								                                         norm='slaney', mel_scale='slaney')
				self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
				
				for param in self.model.parameters():
						param.requires_grad=False				

		def segment_specs(self, x, seg_length, seg_hop=1, max_length=None):
				if seg_length % 2 == 0:
				    raise ValueError('seg_length must be odd! (seg_lenth={})'.format(seg_length))
				if not torch.is_tensor(x):
				    x = torch.tensor(x)

				n_wins = x.shape[2]-(seg_length-1)
				bs = x.shape[0]
				# broadcast magic to segment melspec
				idx1 = torch.arange(seg_length)
				idx2 = torch.arange(n_wins)
				idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
				x = x.transpose(2,1)[:,idx3,:].unsqueeze(2).transpose(4,3)
				if seg_hop>1:
				    x = x[:,::seg_hop,:]
				    n_wins = int(np.ceil(n_wins/seg_hop))
				if max_length is not None:
				    x_padded = torch.zeros((bs, max_length, x.shape[2], x.shape[3], x.shape[4]))
				    x_padded[:,:n_wins,:] = x
				    x = x_padded
				n_wins = [n_wins]*bs
				return x, np.array(n_wins)

		def mos(self, processed_data):
				spec = self.transform(processed_data)
				spec = self.amp_to_db(spec)
				x_spec_seg, n_wins_ = self.segment_specs(spec, self.args['ms_seg_length'], self.args['ms_seg_hop_length'], self.args['ms_max_segments'])
				n_wins = torch.tensor(n_wins_)
				out = self.model(x_spec_seg.to(self.device), n_wins.to(self.device))
				mos = out[:,0]
				return mos				

		def forward(self, processed_data):
				mos = self.mos(processed_data)
				loss = nn.functional.mse_loss(torch.ones_like(mos)*5, mos)
				return loss

if __name__=='__main__':
    loss = NISQALoss(device='cuda')
    inp = torch.rand(1,16000, requires_grad=True)
    inp = inp/torch.max(torch.abs(inp))
    x = inp.clone()
    for i in range(40):
        score = loss.mos(inp)
        inp.retain_grad()
        print(score)
        score.backward(retain_graph=True)
        grads = inp.grad
        inp = inp + grads
    spec_inp = np.abs(librosa.stft(x[0].detach().numpy(), n_fft=512, win_length=512, hop_length=256))
    spec_out = np.abs(librosa.stft(inp[0].detach().numpy(), n_fft=512, win_length=512, hop_length=256))
    inp = inp/torch.max(torch.abs(inp))
    sf.write('inp_noise.wav', x[0].detach().numpy(), 16000)
    sf.write('reconstructed_noise.wav', inp[0].detach().numpy(), 16000)
    '''
    plt.plot(x[0].detach().numpy())
    plt.plot(inp[0].detach().numpy())
    plt.show()
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(np.log10(spec_inp+1e-5), origin='lower', aspect='auto')
    ax[1].imshow(np.log10(spec_out+1e-5), origin='lower', aspect='auto')
    plt.show()
    '''
    '''
    noisy_files = glob.glob('/data/sivaganesh/pv/tvcn_torch/tvcnGAN/ValentiniData/test/noisy/'+'*.wav')
    for noisy_file in noisy_files:
        print(noisy_file)
        data, fs = torchaudio.load(noisy_file)
        out = loss.mos(data)
        print(out)
        break
    '''
    #inp = torch.rand(5, 32000)
    #out = loss(inp)
    #print(out)
