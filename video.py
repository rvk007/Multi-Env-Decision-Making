import os
import sys

import imageio
import cv2
import numpy as np
import torch

import utils


class VideoRecorder(object):
    def __init__(self, root_dir, fps=20):
        self.save_dir = utils.make_dir(root_dir,
                                       'eval_video') if root_dir else None
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array')
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path,
                            self.frames,
                            fps=self.fps,
                            macro_block_size=10)


class TrainVideoRecorder(object):
    def __init__(self, root_dir, aug_trans, fps=20):
        self.save_dir = utils.make_dir(root_dir,
                                       'train_video') if root_dir else None
        self.save_dir_aug = utils.make_dir(
            root_dir, 'train_aug_video') if root_dir else None
        self.fps = fps
        self.aug_trans = aug_trans
        self.frames = []
        self.aug_frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.aug_frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[0],
                               dsize=(512, 512),
                               interpolation=cv2.INTER_CUBIC)
            obs_aug = self.aug_trans(
                torch.as_tensor(obs).float().unsqueeze(0)).numpy()[0].astype(
                    np.uint8)
            frame_aug = cv2.resize(obs_aug[0],
                                   dsize=(512, 512),
                                   interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)
            self.aug_frames.append(frame_aug)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path,
                            self.frames,
                            fps=self.fps,
                            macro_block_size=16)
            path = os.path.join(self.save_dir_aug, file_name)
            imageio.mimsave(path,
                            self.aug_frames,
                            fps=self.fps,
                            macro_block_size=16)