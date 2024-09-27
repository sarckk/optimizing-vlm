from functools import partial
import os
import base64
import io
from typing import List
import pandas as pd
from PIL import Image
import itertools
import torch
import torchvision.transforms as transforms
import datasets

from torch.utils.data import (
    DataLoader,
    IterableDataset,
    DistributedSampler,
)
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

from configs import (
    CURRENT_DIR,
    TrainerConfig,
    TrainDataset,
    NUM_DATA_REPEAT,
    DataConfig,
)


class CharTokenizer:
    def __init__(self, vocab_path: str, pad_token):
        filename = os.path.join(CURRENT_DIR, vocab_path)
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        # Character encoding and decoding functions
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        self.pad_token_id = len(stoi)
        stoi[pad_token] = self.pad_token_id
        self.stoi = stoi

        itos = {i: ch for i, ch in enumerate(chars)}
        itos[self.pad_token_id] = pad_token
        self.itos = itos

        vocab_size = len(stoi.keys())

        self.vocab_size: int = vocab_size

    def encode(self, text: str) -> str:
        return [self.stoi[c] for c in text]

    def decode(self, indices: List[int]) -> str:
        return "".join(self.itos[i] for i in indices)

    def get_vocab_size(self) -> int:
        return self.vocab_size


class InfiniteDistributedSampler(Sampler):
    """
    Generate infinite shuffled indices within len(dataframe)
    such that different ranks get non-overlapping indices

    Adapted from Detectron2 sampler:
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py#L15
    """

    def __init__(self, size):
        self.size = size  # Size of dataframe craeted from .csv
        try:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        except:
            # single-device training
            self.rank = 0
            self.world_size = 1
        self.seed = 0
        self.epoch = 0

    def _infinite_shuffled_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            yield from torch.randperm(self.size, generator=g).tolist()

    def __iter__(self):
        yield from itertools.islice(
            self._infinite_shuffled_indices(), self.rank, None, self.world_size
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Ensure we use a different random ordering for each epoch.
        """
        self.epoch = epoch


class ImageCaptionDataset(IterableDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        img_col: str,
        caption_col: str,
        tokenizer: CharTokenizer,
    ):
        super().__init__()
        self.df = df
        self.img_size = img_size
        self.sampler = InfiniteDistributedSampler(len(self.df))
        self.tokenizer = tokenizer
        self.img_col = img_col
        self.caption_col = caption_col

    def __iter__(self):
        for index in self.sampler:
            row = self.df.iloc[index]
            base64_str = row[self.img_col]
            image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
            caption = row[self.caption_col]
            sample = _preprocess_img_caption_pair(
                self.img_size, self.tokenizer, {"image": [image], "caption": [caption]}
            )
            yield {"image": sample["image"][0], "caption": sample["caption"][0]}


def _preprocess_img_caption_pair(img_size, tokenizer, example):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images = example["image"]
    captions = example["caption"]
    images_t = [transform(img.convert("RGB")).unsqueeze(0) for img in images]
    captions_t = [
        torch.tensor(
            tokenizer.encode(caption.replace("\r", "").replace("\n", "")),
            dtype=torch.long,
        )
        for caption in captions
    ]
    return {"image": images_t, "caption": captions_t}


def pad_caption_indices(padding, max_seq_length, pad_token_id, batch):
    text_indices = [example["caption"] for example in batch]
    images = [example["image"].squeeze() for example in batch]
    device = text_indices[0].device

    bsz = len(batch)

    if padding == "longest":
        max_length = max(len(c) for c in text_indices)
    elif padding == "max_length":
        assert (
            max_seq_length is not None
        ), f"'max_seq_length' arg in trainer config must be a number if padding set to 'max_length'"
        max_length = max_seq_length
    else:
        raise Exception(
            f"padding type of {padding} not valid! valid types are 'longest' or 'max_length'"
        )

    # shape: [bsz, 3, 96, 96]
    images = torch.stack(images, dim=0)

    # shape: [bsz, max_length]
    padded_text = torch.full(
        (bsz, max_length), fill_value=pad_token_id, dtype=torch.long
    ).to(device)

    for i, text in enumerate(text_indices):
        fill_len = min(len(text), max_length)
        padded_text[i, :fill_len] = text[0:fill_len]

    # shape: [bsz, max_length]
    targets = torch.cat(
        [
            padded_text[:, 1:],
            torch.full(
                (bsz, 1),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=1,
    )
    assert (
        padded_text.shape == targets.shape
    ), "Expected src and target sequence to have the same shape"

    return images, padded_text, targets


def _create_csv_dataset(
    img_size: int,
    tokenizer: CharTokenizer,
    data_config: DataConfig,
):
    input_path = os.path.join(CURRENT_DIR, data_config.data_path)
    df = pd.read_csv(input_path)
    df = pd.concat([df] * NUM_DATA_REPEAT)[
        [data_config.image_col, data_config.caption_col]
    ]
    train_split = int(data_config.train_split_ratio * len(df))
    df_train = df.iloc[:train_split]
    df_val = df.iloc[train_split:]

    train_set = ImageCaptionDataset(
        df=df_train,
        img_size=img_size,
        img_col=data_config.image_col,
        caption_col=data_config.caption_col,
        tokenizer=tokenizer,
    )
    val_set = ImageCaptionDataset(
        df=df_val,
        img_size=img_size,
        img_col=data_config.image_col,
        caption_col=data_config.caption_col,
        tokenizer=tokenizer,
    )
    return train_set, val_set


def _create_coco_dataset(
    img_size: int,
    tokenizer: CharTokenizer,
):
    full_dataset = datasets.load_dataset("RIW/small-coco")
    full_dataset = full_dataset.select_columns(["image", "caption"])
    train_ds = full_dataset["train"]

    # Max length of sequence we will see during training is 245
    # print(f"max length: {max([len(e['caption']) for e in train_ds])}")

    train_ds = train_ds.with_transform(
        partial(_preprocess_img_caption_pair, img_size, tokenizer)
    )
    val_ds = full_dataset["validation"]
    val_ds = val_ds.with_transform(
        partial(_preprocess_img_caption_pair, img_size, tokenizer)
    )
    return train_ds, val_ds


def _create_datasets(
    img_size: int,
    tokenizer: CharTokenizer,
    data_config: DataConfig,
):
    dataset: TrainDataset = data_config.dataset

    if dataset == TrainDataset.COCO.value:
        train_set, val_set = _create_coco_dataset(
            img_size,
            tokenizer,
        )
    elif dataset == TrainDataset.CSV.value:
        train_set, val_set = _create_csv_dataset(
            img_size,
            tokenizer,
            data_config,
        )
    else:
        raise Exception(f"Dataset {dataset} not supported!")
    return train_set, val_set


def _create_dataloader(
    dataset, batch_size, pad_token_id, ds_type, use_ddp, padding, max_seq_length
):
    ddp_sampler, train_sampler = None, None
    if ds_type == TrainDataset.COCO.value and use_ddp:
        train_sampler = DistributedSampler(dataset, shuffle=True)
        ddp_sampler = train_sampler
    elif hasattr(dataset, "sampler"):
        train_sampler = dataset.sampler

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,  # transfer to device from pinned memory on host
        shuffle=False,  # DDP requires shuffle=False
        collate_fn=partial(pad_caption_indices, padding, max_seq_length, pad_token_id),
        num_workers=4,  # dataloading on side processes
        sampler=ddp_sampler,
    )
    return dataloader, train_sampler


def get_dataloaders(
    img_size: int,
    trainer_config: TrainerConfig,
    data_config: DataConfig,
):
    tokenizer = CharTokenizer(data_config.vocab_path, data_config.pad_token)

    train_set, val_set = _create_datasets(
        img_size,
        tokenizer,
        data_config,
    )

    train_dataloader, train_sampler = _create_dataloader(
        train_set,
        trainer_config.train_batch_size,
        tokenizer.pad_token_id,
        data_config.dataset,
        trainer_config.use_ddp,
        data_config.padding,
        data_config.max_seq_length,
    )
    val_dataloader, _ = _create_dataloader(
        val_set,
        trainer_config.eval_batch_size,
        tokenizer.pad_token_id,
        data_config.dataset,
        trainer_config.use_ddp,
        data_config.padding,
        data_config.max_seq_length,
    )

    return (tokenizer, train_dataloader, val_dataloader, train_sampler)
