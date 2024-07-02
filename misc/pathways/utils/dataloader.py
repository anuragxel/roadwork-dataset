from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

class SceneDataset(Dataset):
	def __init__(self, data, resize, total_len):
		""" Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""

		self.trajectories, self.meta, self.scene_list = self.split_trajectories_by_scene(data, total_len)
		self.trajectories = self.trajectories * resize

	def __len__(self):
		return len(self.trajectories)

	def __getitem__(self, idx):
		trajectory = self.trajectories[idx]
		meta = self.meta[idx]
		scene = self.scene_list[idx]
		return trajectory, meta, scene

	def split_trajectories_by_scene(self, data, total_len):
		trajectories = []
		meta = []
		scene_list = []
		for meta_id, meta_df in tqdm(data.groupby('sceneId', as_index=False), desc='Prepare Dataset'):
			trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
			meta.append(meta_df)
			scene_list.append(meta_df.iloc()[0:1].sceneId.item())
		return np.array(trajectories, dtype='object'), meta, scene_list


def scene_collate(batch):
	trajectories = []
	meta = []
	scene = []
	for _batch in batch:
		trajectories.append(_batch[0])
		meta.append(_batch[1])
		scene.append(_batch[2])
	return torch.Tensor(trajectories).squeeze(0), meta, scene[0]

import os, json, cv2
import albumentations as A
from segmentation_models_pytorch.encoders import get_preprocessing_fn
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")

class WorkzoneTrajectoryDataset(Dataset):
	def __init__(self, train_image_path, train_data_path, params, total_len, sort_by_curvature=False, clip_by_thresholds = None):
		self.total_len = total_len
		with open(train_data_path, 'r') as f:
			self.data = json.load(f)

		encoder_name = params['segmentation_model_backbone']
		pretrained_weights_name = params["segmentation_pretrained_dataset"]
		resizeW = params["resizeW"]
		resizeH = params["resizeH"]
		resizeW_pad = params["resizeW_pad"]
		resizeH_pad = params["resizeH_pad"]
		origW = params["origW"]
		origH = params["origH"]
		preprocess_input = get_preprocessing_fn(
			encoder_name = encoder_name,
			pretrained = pretrained_weights_name,
		)
		self.transform =  A.Compose([
			A.Resize(width=resizeW, height=resizeH),
			A.Lambda(name = "image_preprocessing", image = preprocess_input),
			A.PadIfNeeded(resizeH_pad, resizeW_pad),
			A.Lambda(name = "to_tensor", image = to_tensor),
		])

		resize_factor_W = resizeW / origW
		resize_factor_H = resizeH / origH

		self.img_paths = []
		self.trajectories = []		
		self.metas = []
		self.curvatures = []
		ix = 0 
		for img_info in self.data:
			ix += 1
			self.img_paths.append(os.path.join(train_image_path, img_info['image']))
			trajectory = np.array([
				np.array([
					# resize the trajectory to fit the image scale
					pos['x'] * resize_factor_W, pos['y'] * resize_factor_H
				]) for pos in img_info['trajectory']
			])
			trajectory = trajectory.reshape(-1, self.total_len, 2)
			self.trajectories.append(trajectory)
			meta = {
				"image_path" : os.path.join(train_image_path, img_info['image']),
				"frame_id" : img_info["id"],
				# "work_objects" : img_info["objects"]
			}
			self.metas.append(meta)
			self.curvatures.append(self.average_curvature_of_path(trajectory))
			## for testing
			# if ix > 200:
			# 	break
	
		if sort_by_curvature:
			sorted_indices = np.argsort(self.curvatures)[::-1]
			self.img_paths = [self.img_paths[i] for i in sorted_indices]
			self.trajectories = [self.trajectories[i] for i in sorted_indices]
			self.metas = [self.metas[i] for i in sorted_indices]
			self.curvatures = [self.curvatures[i] for i in sorted_indices]
		self.clip_by_thresholds = clip_by_thresholds

		if clip_by_thresholds is not None:
			self.clip_dataset(self.clip_by_thresholds)

	def clip_dataset(self, clip_by_thresholds):
		sorted_indices = np.argsort(self.curvatures)
		self.img_paths = [self.img_paths[i] for i in sorted_indices]
		self.trajectories = [self.trajectories[i] for i in sorted_indices]
		self.metas = [self.metas[i] for i in sorted_indices]
		self.curvatures = [self.curvatures[i] for i in sorted_indices]

		low_thresh, high_thresh = clip_by_thresholds
		low_thresh_idx = -1
		high_thresh_idx = -1
		for idx in range(len(self.curvatures)):
			if self.curvatures[idx] > low_thresh:
				low_thresh_idx = idx - 1
				break			
		for idx in range(len(self.curvatures) - 1, -1, -1):
			if self.curvatures[idx] < high_thresh:
				high_thresh_idx = idx + 1
				break
		indices = range(low_thresh_idx, high_thresh_idx)
		self.img_paths = [self.img_paths[i] for i in indices][::-1]
		self.trajectories = [self.trajectories[i] for i in indices][::-1]
		self.metas = [self.metas[i] for i in indices][::-1]
		self.curvatures = [self.curvatures[i] for i in indices][::-1]


	def average_curvature_of_path(self, trajectory):
		## trajectory: numpy array of shape (num_timesteps, 2)
		## returns: average curvature of the path
		trajectory = trajectory.reshape(-1, 2)
		num_timesteps = trajectory.shape[0]

		curvature = 0
		for i in range(1, num_timesteps-1):
			x1, y1 = trajectory[i-1]
			x2, y2 = trajectory[i]
			x3, y3 = trajectory[i+1]
			curvature += np.abs((x1-x2)*(y2-y3) - (y1-y2)*(x2-x3)) / np.sqrt((x1-x2)**2 + (y1-y2)**2)**3

		return curvature / (num_timesteps-2)

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		img_path = self.img_paths[index]
		meta = self.metas[index]
		meta["curvature"] = self.curvatures[index]

		img = cv2.imread(img_path)
		print(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = self.transform(image=img)
		img = img["image"]

		trajectory = self.trajectories[index]
		
		return img, trajectory, meta

def workzone_collate(batch):
	imgs = []
	trajectories = []
	metas = []
	for _batch in batch:
		imgs.append(_batch[0])
		trajectories.append(_batch[1])
		metas.append(_batch[2])
	return torch.Tensor(imgs), torch.Tensor(trajectories), metas

from torch.utils.data import DataLoader
def get_data_loader(dataset_instance, batch_size, collate_fn):
	return DataLoader(dataset_instance, batch_size=batch_size, collate_fn=collate_fn)

if __name__ == "__main__":
	import yaml
	CONFIG_FILE_PATH = "../config/workzone.yaml"
	with open(CONFIG_FILE_PATH) as file:
		params = yaml.load(file, Loader=yaml.FullLoader)
	x = WorkzoneTrajectoryDataset(
		train_data_path="/longdata/anurag_storage/workzone_traj/trajectories_train_equidistant.json",
		train_image_path="/longdata/anurag_storage/workzone_traj/",
		params=params,
		total_len=20,
		sort_by_curvature=True	
	)
	# print(x.curvatures)

	## threshold the curvature into 3 bins: low, medium, high
	# high curvature threshold is one standard deviation above the mean
	high_curvature_threshold = np.mean(x.curvatures) + np.std(x.curvatures)
	# low curvature threshold is one standard deviation below the mean
	low_curvature_threshold = np.mean(x.curvatures) - np.std(x.curvatures)
	# high_curvature_threshold = np.percentile(x.curvatures, 85)
	# low_curvature_threshold = np.percentile(x.curvatures, 15)
	# number of high curvature paths
	print(np.sum(np.array(x.curvatures) > high_curvature_threshold))
	# number of low curvature paths
	print(np.sum(np.array(x.curvatures) < low_curvature_threshold))
	# number of medium curvature paths

	x.clip_dataset([
		# low_curvature_threshold, 
		high_curvature_threshold,
		1000000
	])

	y = get_data_loader(x, 4, collate_fn=workzone_collate)


	for img, trajectory, meta in y:
		print([m["curvature"] for m in meta])
		# print(meta)
		# print(img.shape, trajectory.shape, len(meta))
		# break

	# print(x.__getitem__(0))
	
	# # visualize the first image and trajectory
	# img, trajectory, meta = x.__getitem__(100)
	# from matplotlib import pyplot as plt
	# plt.imshow(img.transpose(1, 2, 0))
	# plt.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'r')
	# plt.savefig("test.png")	
