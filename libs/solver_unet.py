import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from libs.evaluation import *
from models.unet import U_Net
from models.attunet import AttU_Net
from models.r2attunet import R2AttU_Net
from models.r2unet import R2U_Net
import csv
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from tensorboardX import SummaryWriter
from libs.losses import SegLoss
from libs.utils import *


class Solver_unet(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		self.args = config


		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.mseloss = torch.nn.MSELoss(reduce=True, size_average=True)
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.result_path
		self.result_path = config.result_path + "/visualization/"

		if not os.path.exists(self.result_path):
			os.makedirs(self.result_path)

		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t

		self.unet_path = config.check_path
		print(self.unet_path)
		self.build_model()
		self.writter = SummaryWriter(self.model_path)

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.augmentation_prob))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			if self.device == 'cpu':
				self.unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu'))) 
			else:
				self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type, unet_path))
		else:
			if self.args.resume_path != '':
				if self.device == 'cpu':
					self.unet.load_state_dict(torch.load(self.args.resume_path, map_location=torch.device('cpu'))) 
				else:
					self.unet.load_state_dict(torch.load(self.args.resume_path))
				print('%s is Successfully Loaded from %s'%(self.model_type, self.args.resume_path))
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.


			f_vb = open(os.path.join(self.model_path,'validation_blood_res.csv'), 'a', encoding='utf-8', newline='')
			wr_vb = csv.writer(f_vb)
			wr_vb.writerow(['epoch','acc','SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
			f_vb.close()

			f_vc = open(os.path.join(self.model_path,'validation_choroid_res.csv'), 'a', encoding='utf-8', newline='')
			wr_vc = csv.writer(f_vc)
			wr_vc.writerow(['epoch','acc','SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
			f_vc.close()


			f = open(os.path.join(self.model_path,'train_multitask_res.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow(['epoch','acc_blood','SE_blood', 'SP_blood', 'PC_blood', 'F1_blood', 'JS_blood', 'DC_blood', 'acc_choroid','SE_choroid', 'SP_choroid', 'PC_choroid', 'F1_choroid', 'JS_choroid', 'DC_choroid'])

			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient

				acc_choroid = 0.	# Accuracy
				SE_choroid = 0.		# Sensitivity (Recall)
				SP_choroid = 0.		# Specificity
				PC_choroid = 0. 	# Precision
				F1_choroid = 0.		# F1 Score
				JS_choroid = 0.		# Jaccard Similarity
				DC_choroid = 0.		# Dice Coefficient
				length = 0

				loss_blood_save = 0
				loss_choroid_save = 0
				loss_adr_save = 0
				loss_save = 0

				print("Training Epoch " + str(epoch + 1) + "...............")
				for i, (images, GTs, filename) in enumerate(self.train_loader):
					# GT : Ground Truth
					
					images = images.to(self.device)
					if self.args.type == 'blood':
						GT_blood = GTs
						GT_blood = GT_blood.to(self.device) #b, h, w
						SRs, _, _, _ = self.unet(images)
						[SR_blood] = SRs
						SR_blood_probs = F.softmax(SR_blood, dim=1)
						RS_blood = torch.argmax(SR_blood_probs, dim=1)
						loss_blood = SegLoss(SR_blood_probs, GT_blood)
						loss_choroid = torch.zeros(1).to(self.device)
					elif self.args.type == 'choroid':
						GT_choroid = GTs
						GT_choroid = GT_choroid.to(self.device)
						SRs, _, _, _ = self.unet(images)
						[SR_choroid] = SRs
						SR_choroid_probs = F.softmax(SR_choroid, dim=1)
						RS_choroid = torch.argmax(SR_choroid_probs, dim=1)	
						loss_choroid = SegLoss(SR_choroid_probs, GT_choroid)
						loss_blood = torch.zeros(1).to(self.device)
					else:
						[GT_blood, GT_choroid] = GTs
						GT_blood = GT_blood.to(self.device) #b, h, w
						GT_choroid = GT_choroid.to(self.device)
						SRs, _, _, _ = self.unet(images)
						[SR_blood, SR_choroid] = SRs
						SR_choroid_probs = F.softmax(SR_choroid, dim=1)
						SR_blood_probs = F.softmax(SR_blood, dim=1)
						RS_blood = torch.argmax(SR_blood_probs, dim=1)
						RS_choroid = torch.argmax(SR_choroid_probs, dim=1)	
						loss_choroid = SegLoss(SR_choroid_probs, GT_choroid)
						loss_blood = SegLoss(SR_blood_probs, GT_blood)

					loss_adr = torch.zeros(1).to(self.device)
					loss = loss_blood + loss_choroid + loss_adr * self.args.lamda

					epoch_loss += loss.item()
					loss_save += loss.item()
					loss_choroid_save += loss_choroid.item()
					loss_blood_save += loss_blood.item()
					loss_adr_save += loss_adr.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()
					length += 1

					##Eval
					if self.args.type != 'choroid':
						acc += get_accuracy_ml(RS_blood, GT_blood)
						SE += get_sensitivity_ml(RS_blood, GT_blood)
						SP += get_specificity_ml(RS_blood, GT_blood)
						PC += get_precision_ml(RS_blood, GT_blood)
						F1 += get_F1_ml(RS_blood, GT_blood)
						JS += get_JS_ml(RS_blood, GT_blood)
						DC += get_DC_ml(RS_blood, GT_blood)

					if self.args.type != 'blood':
						acc_choroid += get_accuracy_ml(RS_choroid, GT_choroid)
						SE_choroid += get_sensitivity_ml(RS_choroid, GT_choroid)
						SP_choroid += get_specificity_ml(RS_choroid, GT_choroid)
						PC_choroid += get_precision_ml(RS_choroid, GT_choroid)
						F1_choroid += get_F1_ml(RS_choroid, GT_choroid)
						JS_choroid += get_JS_ml(RS_choroid, GT_choroid)
						DC_choroid += get_DC_ml(RS_choroid, GT_choroid)

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length

				acc_choroid = acc_choroid/length
				SE_choroid = SE_choroid/length
				SP_choroid = SP_choroid/length
				PC_choroid = PC_choroid/length
				F1_choroid = F1_choroid/length
				JS_choroid = JS_choroid/length
				DC_choroid = DC_choroid/length

				print('Epoch [%d/%d], Loss: %.4f, Loss_cls: %.4f, Loss_sa: %.4f, Loss_adr: %.4f, \n[Training] [Blood] Loss: %.4f, SE: %.4f, PC: %.4f, F1: %.4f, DC: %.4f \n[Training] [Choroid]  Loss: %.4f, SE: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
				epoch+1, self.num_epochs, \
				loss_save / (i+1), loss_blood_save / (i+1), loss_choroid_save / (i+1), loss_adr_save / (i+1), \
				loss_blood_save / (i+1), SE,PC,F1,DC, \
				loss_choroid_save / (i+1), SE_choroid, PC_choroid, F1_choroid, DC_choroid))
				wr.writerow([epoch+1, acc, SE, SP, PC, F1, JS, DC, SE_choroid, SP_choroid, PC_choroid, F1_choroid, JS_choroid, DC_choroid])
				self.writter.add_scalar('Loss/Loss', loss_save / (i+1), epoch + 1)
				self.writter.add_scalar('Loss/Loss_Choroid', loss_choroid_save / (i+1), epoch + 1)
				self.writter.add_scalar('Loss/Loss_Blood', loss_blood_save  / (i+1), epoch + 1)
				self.writter.add_scalar('Loss/Loss_Adaptive', loss_adr_save  / (i+1), epoch + 1)
				self.writter.add_scalar('Blood/Dice', DC, epoch + 1)
				self.writter.add_scalar('Blood/Recall', SE, epoch + 1)
				self.writter.add_scalar('Blood/Precision', PC, epoch + 1)
				self.writter.add_scalar('Choroid/Dice', DC_choroid, epoch + 1)
				self.writter.add_scalar('Choroid/Recall', SE_choroid, epoch + 1)
				self.writter.add_scalar('Choroid/Precision', PC_choroid, epoch + 1)

				if (epoch + 1) in self.num_epochs_decay:
					lr = lr / 10.0
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				if (epoch + 1) % self.args.eval_frequency == 0:
					unet_score = self.validation(epoch + 1)

					# Save Best U-Net model
					if unet_score > best_unet_score:
						best_unet_score = unet_score
						best_epoch = epoch
						best_unet = self.unet.state_dict()
						print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
						torch.save(best_unet,unet_path)
						self.check_path = unet_path
			self.validation(epoch + 1)
			final_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f_final.pkl' %(self.model_type,self.num_epochs,self.lr,self.augmentation_prob))
			torch.save(self.unet.state_dict(), final_path)
	
	def validation(self, epoch):

		self.unet.train(False)
		self.unet.eval()

		tmp  = self.args.type
		if self.args.is_continuely:
			print("Using Multitask Learning Validation.....")
			self.args.type = 'multitask'

		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient

		acc_choroid = 0.	# Accuracy
		SE_choroid = 0.		# Sensitivity (Recall)
		SP_choroid = 0.		# Specificity
		PC_choroid = 0. 	# Precision
		F1_choroid = 0.		# F1 Score
		JS_choroid = 0.		# Jaccard Similarity
		DC_choroid = 0.		# Dice Coefficient
		length=0

		loss_blood_save = 0
		loss_choroid_save = 0
		loss_adr_save = 0
		loss_save = 0

		print("Validation Epoch " + str(epoch + 1) + "...............")
		for i, (images, GTs, filename)  in enumerate(self.valid_loader):


			if (i+1) % 20 == 0:
				print('Image [%d/%d]' % (i+1, len(self.train_loader)))

			images = images.to(self.device)
			with torch.no_grad():
				if self.args.type == 'blood':
					GT_blood = GTs
					GT_blood = GT_blood.to(self.device) #b, h, w
					SRs, fs, _, _ = self.unet(images)
					[SR_blood] = SRs
					SR_blood_probs = F.softmax(SR_blood, dim=1)
					RS_blood = torch.argmax(SR_blood_probs, dim=1)
					loss_blood = SegLoss(SR_blood_probs, GT_blood)
					loss_choroid = torch.zeros(1).to(self.device)
				elif self.args.type == 'choroid':
					GT_choroid = GTs
					GT_choroid = GT_choroid.to(self.device)
					SRs, fs, _, _ = self.unet(images)
					[SR_choroid] = SRs
					SR_choroid_probs = F.softmax(SR_choroid, dim=1)
					RS_choroid = torch.argmax(SR_choroid_probs, dim=1)	
					loss_choroid = SegLoss(SR_choroid_probs, GT_choroid)
					loss_blood = torch.zeros(1).to(self.device)
				else:
					[GT_blood, GT_choroid] = GTs
					GT_blood = GT_blood.to(self.device) #b, h, w
					GT_choroid = GT_choroid.to(self.device)
					SRs, fs, _, _ = self.unet(images)
					[SR_blood, SR_choroid] = SRs
					SR_choroid_probs = F.softmax(SR_choroid, dim=1)
					SR_blood_probs = F.softmax(SR_blood, dim=1)
					RS_blood = torch.argmax(SR_blood_probs, dim=1)
					RS_choroid = torch.argmax(SR_choroid_probs, dim=1)	
					loss_choroid = SegLoss(SR_choroid_probs, GT_choroid)
					loss_blood = SegLoss(SR_blood_probs, GT_blood)

				loss_adr = torch.zeros(1).to(self.device)
				loss = loss_blood + loss_choroid + loss_adr * self.args.lamda

				#epoch_loss += loss.item()
				loss_save += loss.item()
				loss_choroid_save += loss_choroid.item()
				loss_blood_save += loss_blood.item()
				loss_adr_save += loss_adr.item()

				##Eval
				if self.args.type != 'choroid':
					acc += get_accuracy_ml(RS_blood, GT_blood)
					SE += get_sensitivity_ml(RS_blood, GT_blood)
					SP += get_specificity_ml(RS_blood, GT_blood)
					PC += get_precision_ml(RS_blood, GT_blood)
					F1 += get_F1_ml(RS_blood, GT_blood)
					JS += get_JS_ml(RS_blood, GT_blood)
					DC += get_DC_ml(RS_blood, GT_blood)

				if self.args.type != 'blood':
					acc_choroid += get_accuracy_ml(RS_choroid, GT_choroid)
					SE_choroid += get_sensitivity_ml(RS_choroid, GT_choroid)
					SP_choroid += get_specificity_ml(RS_choroid, GT_choroid)
					PC_choroid += get_precision_ml(RS_choroid, GT_choroid)
					F1_choroid += get_F1_ml(RS_choroid, GT_choroid)
					JS_choroid += get_JS_ml(RS_choroid, GT_choroid)
					DC_choroid += get_DC_ml(RS_choroid, GT_choroid)

			length += 1

			cur_save_path, f_path, rs_path, gd_path, gd_blood_path, gd_choroid_path, rs_blood_path, rs_choroid_path = mk_save_dir_val(self.result_path, epoch)

			b = images.shape[0]
			for cur_b in range(0, b):

				###save features####
				#save_features(fs[cur_b, :, :, :], f_path, filename[cur_b], type='global')
				#save_features(fd_c[cur_b, :, :, :], f_path, filename[cur_b], type='choroid')
				#save_features(fd_b[cur_b, :, :, :], f_path, filename[cur_b], type='blood')

				###save results####
				cur_img = tensor2image(images[cur_b, :, :, :])
				#save_segmentation(GT_blood[cur_b, :, :].cpu().data, cur_img, gd_path, filename[cur_b], 'gt')
				if self.args.type != 'choroid':
					save_segmentation(RS_blood[cur_b, :, :].cpu().data, cur_img, rs_blood_path, filename[cur_b], 'blood')
					save_features(fs[cur_b, :, :, :], f_path, filename[cur_b], type='blood')
				elif self.args.type != 'blood':
					save_segmentation(RS_choroid[cur_b, :, :].cpu().data, cur_img, rs_choroid_path, filename[cur_b], 'choroid')
					save_features(fs[cur_b, :, :, :], f_path, filename[cur_b], type='choroid')


		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length
		
		unet_score = F1


		acc_choroid = acc_choroid/length
		SE_choroid = SE_choroid/length
		SP_choroid = SP_choroid/length
		PC_choroid = PC_choroid/length
		F1_choroid = F1_choroid/length
		JS_choroid = JS_choroid/length
		DC_choroid = DC_choroid/length
		unet_score = unet_score + F1_choroid

		print('[Validation] [Blood] SE: %.4f, PC: %.4f, F1: %.4f, DC: %.4f \n[Validationning] [Choroid] SE: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
		SE,PC,F1,DC, \
		SE_choroid, PC_choroid, F1_choroid, SE_choroid))
		f = open(os.path.join(self.model_path,'validation_blood_res.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow([epoch+1, acc, SE, SP, PC, F1, JS, DC])
		f.close()

		f2 = open(os.path.join(self.model_path,'validation_choroid_res.csv'), 'a', encoding='utf-8', newline='')
		wr2 = csv.writer(f2)
		wr2.writerow([epoch+1, acc_choroid, SE_choroid, SP_choroid, PC_choroid, F1_choroid, JS_choroid, DC_choroid])
		f2.close()

		self.writter.add_scalar('Test/Blood/Dice', DC, epoch + 1)
		self.writter.add_scalar('Test/Blood/Recall', SE, epoch + 1)
		self.writter.add_scalar('Test/Blood/Precision', PC, epoch + 1)
		self.writter.add_scalar('Test/Blood/ACC', acc, epoch + 1)
		self.writter.add_scalar('Test/Blood/JS', JS, epoch + 1)
		self.writter.add_scalar('Test/Blood/F1', F1, epoch + 1)
		self.writter.add_scalar('Test/Blood/SP', SP, epoch + 1)

		self.writter.add_scalar('Test/Loss/Total', loss_save / length, epoch + 1)
		self.writter.add_scalar('Test/Loss/Choroid', loss_choroid_save / length, epoch + 1)
		self.writter.add_scalar('Test/Loss/Blood', loss_blood_save  / length, epoch + 1)
		self.writter.add_scalar('Test/Loss/Adaptive', loss_adr_save  / length, epoch + 1)


		self.writter.add_scalar('Test/Choroid/Dice', DC_choroid, epoch + 1)
		self.writter.add_scalar('Test/Choroid/Recall', SE_choroid, epoch + 1)
		self.writter.add_scalar('Test/Choroid/Precision', PC_choroid, epoch + 1)
		self.writter.add_scalar('Test/Choroid/ACC', acc_choroid, epoch + 1)
		self.writter.add_scalar('Test/Choroid/JS', JS_choroid, epoch + 1)
		self.writter.add_scalar('Test/Choroid/F1', F1_choroid, epoch + 1)
		self.writter.add_scalar('Test/Choroid/SP', SP_choroid, epoch + 1)

		self.args.type = tmp

		return unet_score
		

	def test(self):

		self.build_model()

		self.unet.load_state_dict(torch.load(self.unet_path, map_location=torch.device('cpu'))) 
		#self.unet.load_state_dict(torch.load(self.unet_path))
		
		self.unet.train(False)
		self.unet.eval()

		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient

		acc_choroid = 0.	# Accuracy
		SE_choroid = 0.		# Sensitivity (Recall)
		SP_choroid = 0.		# Specificity
		PC_choroid = 0. 	# Precision
		F1_choroid = 0.		# F1 Score
		JS_choroid = 0.		# Jaccard Similarity
		DC_choroid = 0.		# Dice Coefficient
		length=0

		print("Test..............")
		for i, (images, GTs, filename)  in enumerate(self.valid_loader):

			if (i+1) % 20 == 0:
				print('Image [%d/%d]' % (i+1, len(self.train_loader)))

			images = images.to(self.device)
			with torch.no_grad():
				if self.args.type == 'blood':
					GT_blood = GTs
					GT_blood = GT_blood.to(self.device) #b, h, w
					SRs, fs, _, _ = self.unet(images)
					[SR_blood] = SRs
					SR_blood_probs = F.softmax(SR_blood, dim=1)
					RS_blood = torch.argmax(SR_blood_probs, dim=1)
					loss_blood = SegLoss(SR_blood_probs, GT_blood)
					loss_choroid = torch.zeros(1).to(self.device)
				elif self.args.type == 'choroid':
					GT_choroid = GTs
					GT_choroid = GT_choroid.to(self.device)
					SRs, fs, _, _ = self.unet(images)
					[SR_choroid] = SRs
					SR_choroid_probs = F.softmax(SR_choroid, dim=1)
					RS_choroid = torch.argmax(SR_choroid_probs, dim=1)	
				else:
					[GT_blood, GT_choroid] = GTs
					GT_blood = GT_blood.to(self.device) #b, h, w
					GT_choroid = GT_choroid.to(self.device)
					SRs, fs, _, _ = self.unet(images)
					[SR_blood, SR_choroid] = SRs
					SR_choroid_probs = F.softmax(SR_choroid, dim=1)
					SR_blood_probs = F.softmax(SR_blood, dim=1)
					RS_blood = torch.argmax(SR_blood_probs, dim=1)
					RS_choroid = torch.argmax(SR_choroid_probs, dim=1)	

				##Eval
				if self.args.type != 'choroid':
					acc += get_accuracy_ml(RS_blood, GT_blood)
					SE += get_sensitivity_ml(RS_blood, GT_blood)
					SP += get_specificity_ml(RS_blood, GT_blood)
					PC += get_precision_ml(RS_blood, GT_blood)
					F1 += get_F1_ml(RS_blood, GT_blood)
					JS += get_JS_ml(RS_blood, GT_blood)
					DC += get_DC_ml(RS_blood, GT_blood)

				if self.args.type != 'blood':
					acc_choroid += get_accuracy_ml(RS_choroid, GT_choroid)
					SE_choroid += get_sensitivity_ml(RS_choroid, GT_choroid)
					SP_choroid += get_specificity_ml(RS_choroid, GT_choroid)
					PC_choroid += get_precision_ml(RS_choroid, GT_choroid)
					F1_choroid += get_F1_ml(RS_choroid, GT_choroid)
					JS_choroid += get_JS_ml(RS_choroid, GT_choroid)
					DC_choroid += get_DC_ml(RS_choroid, GT_choroid)

			length += 1

			cur_save_path = self.result_path
			if not os.path.exists(cur_save_path):
				os.makedirs(cur_save_path)
			
			f_path = cur_save_path + "labels"
			if not os.path.exists(f_path):
				os.makedirs(f_path)

			rs_path = cur_save_path + "/vis_result"
			if not os.path.exists(rs_path):
				os.makedirs(rs_path)

			b, c, h, w = d2.size()

			save_feature_path = f_path + "feature"
			lb_blood_path = f_path + "/blood"
			lb_choroid_path = f_path + "/choroid"
			if not os.path.exists(lb_blood_path):
				os.makedirs(lb_blood_path)

			if not os.path.exists(lb_choroid_path):
				os.makedirs(lb_choroid_path)

			if not os.path.exists(save_feature_path):
				os.makedirs(save_feature_path)

			


			feature_path = rs_path + "/feature"
			rs_blood_path = rs_path + "/blood"
			rs_choroid_path = rs_path + "/choroid"
			if not os.path.exists(rs_blood_path):
				os.makedirs(rs_blood_path)

			if not os.path.exists(feature_path):
				os.makedirs(feature_path)

			if not os.path.exists(rs_choroid_path):
				os.makedirs(rs_choroid_path)

			b = images.shape[0]
			for cur_b in range(0, b):

				###save features####
				#save_features(fs[cur_b, :, :, :], f_path, filename[cur_b], type='global')
				#save_features(fd_c[cur_b, :, :, :], f_path, filename[cur_b], type='choroid')
				#save_features(fd_b[cur_b, :, :, :], f_path, filename[cur_b], type='blood')

				###save results####
				cur_img = tensor2image(images[cur_b, :, :, :])
				#save_segmentation(GT_blood[cur_b, :, :].cpu().data, cur_img, gd_path, filename[cur_b], 'gt')
				if self.args.type != 'choroid':
					save_segmentation(RS_blood[cur_b, :, :].cpu().data, cur_img, rs_blood_path, filename[cur_b], 'blood')
					save_features(fs[cur_b, :, :, :], feature_path, filename[cur_b], type='blood')
					np.save(lb_blood_path + "/" + filename[cur_b], np.int32(RS_blood[cur_b, :, :].cpu().data))
					np.save(save_feature_path + "/" + filename[cur_b] + "_blood", fs[cur_b, :, :, :])
				elif self.args.type != 'blood':
					save_segmentation(RS_choroid[cur_b, :, :].cpu().data, cur_img, rs_choroid_path, filename[cur_b], 'choroid')
					save_features(fs[cur_b, :, :, :], feature_path, filename[cur_b], type='choroid')
					np.save(lb_choroid_path + "/" + filename[cur_b], np.int32(RS_choroid[cur_b, :, :].cpu().data))
					np.save(save_feature_path + "/" + filename[cur_b] + "_choroid", fs[cur_b, :, :, :])



		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length

		acc_choroid = acc_choroid/length
		SE_choroid = SE_choroid/length
		SP_choroid = SP_choroid/length
		PC_choroid = PC_choroid/length
		F1_choroid = F1_choroid/length
		JS_choroid = JS_choroid/length
		DC_choroid = DC_choroid/length


		f = open(os.path.join(self.result_path,'result_blood.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow(['model','acc','SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
		wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC])
		f.close()

		f = open(os.path.join(self.result_path,'result_choroid.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow(['model','acc','SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
		wr.writerow([self.model_type,acc_chorid,SE_chorid,SP_chorid,PC_chorid,F1_chorid,JS_chorid,DC_chorid])
		f.close()




	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=self.output_ch)
			#param_list = list(self.unet.parameters())
		elif self.model_type == 'AttU_Net':
			self.unet = AttU_Net(img_ch=3, output_ch=self.output_ch)
			#param_list = list(self.unet.parameters())
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=3, output_ch=self.output_ch)
			#param_list = list(self.unet.parameters())
		elif self.model_type == 'R2U_Net':
			self.unet = R2U_Net(img_ch=3, output_ch=self.output_ch)
			#param_list = list(self.unet.parameters())
			
		#init_weights(self.unet, 'normal')
		param_list = list(self.unet.parameters())

		print(self.unet)
		
		if self.args.optim == 'adam':
			self.optimizer = optim.Adam(param_list,
										self.lr, [self.beta1, self.beta2])
		else:
			self.optimizer = optim.SGD(param_list, self.lr, momentum=0.9, weight_decay=1e-4)
		self.unet.to(self.device)
	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img