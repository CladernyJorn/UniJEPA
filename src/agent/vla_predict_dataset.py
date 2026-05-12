import json
import os
from random import shuffle

import numpy as np
from functools import partial

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image


def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


class DataProvider(Dataset):
    # need to read whole dataset into memory before loading to gpu
    # predict future 10 steps image
    def __init__(self, dataset_path, image_size=224, future_step=10, use_depth=False):
        self.dataset_path = dataset_path
        self.transform = image_transform
        self.image_size = image_size
        self.episodes, self.instructions, (self.actions,
                                           self.states) = load_preprocessed_data(dataset_path)  # actions are not used
        self.length_episodes = np.cumsum([len(i) for i in self.episodes])
        self.length_episodes = {i: self.length_episodes[i] for i in range(len(self.length_episodes))}
        self.future_step = future_step
        self.use_depth = use_depth
        print("Formatting Future prediction (T2I) data")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        data_dict = self.get_raw_items(index)
        future_index = self.get_future_index(index, future_step=self.future_step)
        data_dict_future = self.get_raw_items(future_index)
        # assert data_dict['input_ids'] == data_dict_future['input_ids']
        # data_dict['images_static_future'] = data_dict_future['images_static']
        # data_dict['images_gripper_future'] = data_dict_future['images_gripper']
        data_dict['observation']['image_primary_future'] = data_dict_future['observation']['image_primary']
        if index == future_index:
            actions = torch.tensor(self.actions[index:future_index + 1])
        else:
            actions = torch.tensor(self.actions[index:future_index])  # n,7
        if actions.shape[0] < self.future_step:
            offset = self.future_step - actions.shape[0]
            pad_tube = torch.zeros(size=(offset, actions.shape[-1]), dtype=actions.dtype)
            pad_tube[:, -1] = actions[-1, -1]  # gripper state of last action is repeated
            actions = torch.cat([actions, pad_tube], dim=0)
        data_dict['action'] = actions  # (self.future_step, 7) (10,7)
        return data_dict

    def get_raw_items(self, index):
        episode_idx, idx = self.get_episode_idx(index)
        episode = self.episodes[episode_idx]
        # sequence_length * epi[0],epi[1],...
        if 'calvin' in self.dataset_path:
            frame_data = np.load(episode[idx], allow_pickle=True)
            image_static = frame_data['rgb_static']  #hwc 0-255
            # image_gripper = np.load(episode[idx], allow_pickle=True)['rgb_gripper']  # hwc,255
            image_static = self.transform(
                Image.fromarray(np.uint8(image_static)), resolution=self.image_size, normalize=False)  # 3,224,224; 0-1
            proprio = frame_data['robot_obs'][np.newaxis, :]  # 1,15
            image_obs = (255 * image_static).unsqueeze(0).to(torch.uint8).permute(0, 2, 3, 1)  # 1,224,224,3
            # image_gripper = self.transform(Image.fromarray(np.uint8(image_gripper)), resolution=self.image_size,normalize=False)
        elif 'real_panda' in self.dataset_path:
            # image_gripper = Image.open(
            #     "/mnt/panda_real_data_processed/2024-04-19-pick_carrot_2obj/episode0000000/color_wrist_1_0001.jpg"
            # ).convert('RGB')
            image_gripper = Image.open(episode[idx]['rgb_gripper']).convert('RGB')  # hwc,255
            image_gripper = self.transform(image_gripper, resolution=self.image_size, normalize=False)  # 3,224,224; 0-1
            proprio = torch.tensor(self.states[index])  # 1,15
            image_obs = (255 * image_gripper).unsqueeze(0).to(torch.uint8).permute(0, 2, 3, 1)  # 1,224,224,3
            # image_static = self.transform(image_static, resolution=self.image_size)
            # image_gripper = self.transform(image_gripper, resolution=self.image_size)
        # elif 'bridge' in self.dataset_path:
        #     image_static = Image.open(episode[idx]['rgb_static']).convert('RGB')
        #     image_gripper = Image.open(episode[idx]['rgb_gripper']).convert('RGB')  # hwc,255
        #     image_static = self.transform(image_static, resolution=self.image_size)
        #     image_gripper = self.transform(image_gripper, resolution=self.image_size)
        else:
            raise NotImplementedError
            # if sequence[0][0].startswith('/'):
            #     patch_images_wrist = [Image.open(epi_path[0]).convert('RGB') for epi_path in sequence]
            #     patch_images_third = [Image.open(epi_path[1]).convert('RGB') for epi_path in sequence]
            # patch_images = (patch_images_wrist, patch_images_third)
        instruction = self.instructions[episode_idx]
        data_dict = dict(
            lang_text=instruction,
            observation={
                'image_primary': image_obs,
                'proprio': proprio
            },
        )
        if self.use_depth:
            data_dict['observation']['image_depth'] = torch.tensor(frame_data['depth_static'])
        return data_dict

    def get_episode_idx(self, index):
        for i, x in self.length_episodes.items():
            if index < x:
                episode_idx = i
                idx = index - self.length_episodes[episode_idx - 1] if i != 0 else index
                return episode_idx, idx
        raise ValueError(f"Index {index} out of range")

    def get_future_index(self, index, future_step=10):
        for i, x in self.length_episodes.items():
            if index < x:
                if index + future_step < x:
                    return index + future_step  # future index is in the same episode
                else:
                    return self.length_episodes[i] - 1  # future index is in the next episode, use the last frame
        raise ValueError(f"Index {index} out of range")


def load_preprocessed_data(dataset_path):
    episodes = []
    instructions = []
    actions = []
    states = []
    assert "processed" in dataset_path
    with open(os.path.join(dataset_path, 'dataset_info.json'), 'r') as f:
        dataset = json.load(f)
    for epi in dataset:
        frames = []
        for frame in epi["frames"]:
            if 'calvin' in dataset_path:
                path_head = "/localssd/data/calvin/task_ABC_D"
                # path_head="/cephfs/shared/hyc/data/calvin/task_ABC_D"
                # path_head = "/localssd/data_calvin/task_ABC_D"
                # path_head = "/localssd/data/calvin/task_ABC_D"
                path_tale = frame["dir"].split("/")  # training/0000000.npz
                frames.append(f"{path_head}/{path_tale[-2]}/{path_tale[-1]}")
                # frames.append(frame["dir"])
                actions.append(frame["rel_action"])
            elif 'real_panda' in dataset_path:
                path_head = "/mnt/"
                # path_head = "/cephfs/shared/processed_data"
                path_tale1 = "/".join(frame["wrist_1"].split("/")[3:])  # /cephfs/shared
                # path_tale2 = "/".join(frame["wrist_2"].split("/")[3:])  # /cephfs/shared
                # frames.append({'rgb_gripper': path_head+path_tale1, 'rgb_static': path_head+path_tale2})
                frames.append({'rgb_gripper': path_head + path_tale1})
                actions.append(frame["action"])
                states.append(frame["state"])
            elif 'bridge' in dataset_path:
                # path_head = "/localssd/data/"
                path_head = dataset_path + "/"  # /localssd/data/bridge_processed/
                path_tale1 = frame["dir"]
                # only have 1 view, used for two
                # frames.append({'rgb_gripper': path_head+path_tale1, 'rgb_static': path_head+path_tale2})
                frames.append({'rgb_gripper': path_head + path_tale1, 'rgb_static': path_head + path_tale1})
                actions.append(frame["action"])
            else:
                raise NotImplementedError

        instructions.append(epi["instruction"])
        episodes.append(frames)
    return episodes, instructions, (actions, states)


def collate_fn(instances,):
    input_ids = [instance["input_ids"] for instance in instances]
    batch = dict(input_ids=input_ids,)
    batch['images_static'] = torch.stack([instance['images_static'] for instance in instances])
    batch['images_gripper'] = torch.stack([instance['images_gripper'] for instance in instances])
    batch['images_static_future'] = torch.stack([instance['images_static_future'] for instance in instances])
    batch['images_gripper_future'] = torch.stack([instance['images_gripper_future'] for instance in instances])
    batch['actions'] = torch.stack([instance['actions'] for instance in instances])
    return batch


def get_vla_predict_data_loader(
    config,
    train=True,
):
    if train:
        dataset_dir = config.dataset_dir
        num_workers = config.num_workers
    else:
        dataset_dir = config.dataset_dir + "_validation"
        num_workers = 0
    if config.use_depth:
        print("Use depth data")
    dataset = DataProvider(dataset_path=dataset_dir, future_step=config.future_step, use_depth=config.use_depth)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=num_workers,
    )

    return dataloader
