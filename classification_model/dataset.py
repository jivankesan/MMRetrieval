# Updated dataset.py file
import torch
import torch.utils.data as data_utl
from classes import class_conversion
import numpy as np
from random import randint
import os
import os.path
import glob
import cv2


class Bal_Dict():

    def __init__(self):
        self.balanced_dict = {}
        self.total_dict = {}

    def bal_update(self, outputs, y_true):

        l =len(y_true)
        y_pred = torch.max(outputs, dim=1)[1]
        y_pred = y_pred.squeeze().tolist()
        y_true = y_true.squeeze().tolist()

        for i in range(l):
            if y_true[i] in self.balanced_dict.keys():
                self.balanced_dict[y_true[i]] += float(y_true[i]==y_pred[i])
            else:
                self.balanced_dict[y_true[i]] = (y_true[i] == y_pred[i])
            if y_true[i] in self.total_dict.keys():
                self.total_dict[y_true[i]] += 1.0
            else:
                self.total_dict[y_true[i]] = 1.0

    def bal_score(self):
        count = 0.0
        for i in self.total_dict.keys():
            if i in self.balanced_dict.keys():
                count += ((1.0*self.balanced_dict[i])/self.total_dict[i])

        return count / len(self.total_dict.keys())


def calculate_accuracy(outputs, targets):

    with torch.no_grad():
        batch_size = targets.size(0)
        pred = torch.max(outputs, dim=1)[1]
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(frames):

  images = []
  for i in frames:
    img=cv2.imread(i)[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    images.append(img)

  images = np.asarray(images, dtype=np.float32)
  images = (images/127.5) - 1

  return images


def load_flow_frames(image_dir, vid, start, num):

  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)

  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, protocol='CS'):
    dataset = []
    num_classes = 31 if protocol == 'CS' else 19

    with open(split_file, 'r') as f:
        data = [os.path.splitext(i.strip())[0] for i in f.readlines()]

    for vid in data:
        video_path = os.path.join(root, vid + ".mp4")
        
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue

        # Open the video file and count the number of frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            continue

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if mode == 'flow':
            num_frames = num_frames // 2
        
        label = np.eye(num_classes)[class_conversion(vid.split('_')[0]) - 1]
        dataset.append((vid, label, num_frames))
    
    return dataset


class Dataset(data_utl.Dataset):
    def __init__(self, split_file, split, root, mode, transforms=None, protocol='CS'):
        self.data = make_dataset(split_file, split, root, mode, protocol)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.sample_duration = 64
        self.step = 2

    def __getitem__(self, index):
        vid, label, nf = self.data[index]
        frame_indices = []
        images = sorted(glob.glob(self.root + vid + "/*"))
        n_frames = len(images)

        if n_frames > self.sample_duration * self.step:
            start = randint(0, n_frames - self.sample_duration * self.step)
            for i in range(start, start + self.sample_duration * self.step, self.step):
                frame_indices.append(images[i])
        elif n_frames < self.sample_duration:
            while len(frame_indices) < self.sample_duration:
                frame_indices.extend(images)
            frame_indices = frame_indices[:self.sample_duration]
        else:
            start = randint(0, n_frames - self.sample_duration)
            for i in range(start, start + self.sample_duration):
                frame_indices.append(images[i])

        if self.mode == 'rgb':
            imgs = load_rgb_frames(frame_indices)
        else:
            imgs = load_flow_frames(self.root, vid, start, 64)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)