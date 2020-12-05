import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os

from util.data_utils import load_data_from_file
# from tokenizer import encode_sequences

"""
Reference
* https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
"""


class CoraDataset(Dataset):
    """Cora dataset for Graph Convolution."""

    def __init__(self, txt_files, root_dir, transform=None):
        """
        Args:
            txt_files (string): txt 파일의 경로
            root_dir (string): 모든 xml 파일이 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.datasets = load_data_from_file(txt_files)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        :return: 데이터셋의 크기
        """
        return len(self.datasets)

    def __getitem__(self, idx):
        """
        i번째 샘플을 찾는데 사용
        :param idx:
        :return: i번째 샘플
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = (self.input_seq[idx], self.output_seq[idx])
        sample = self.datasets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


# ---- TEST ---- #

def show_text_batch(sample_batched):
    """Show texts for a batch of samples."""
    batch_size = len(sample_batched)

    for i in range(batch_size):
        print(sample_batched[:5])


if __name__ == '__main__':

    fr_en_dataset = CoraDataset(txt_files='../data/cora/',
                                root_dir='../data/cora/')

    for i in range(len(fr_en_dataset)):
        sample = fr_en_dataset[i]

        print(i, sample.shape)
        print(sample[:5])

        if i == 3:
            break

    dataloader = DataLoader(fr_en_dataset, batch_size=4,
                            shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.size())

        # observe 4th batch and stop.
        if i_batch == 3:
            show_text_batch(sample_batched)
            break
