import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    audios = [item["audio"] for item in dataset_items]
    labels = [item["labels"] for item in dataset_items]

    padded_audio = pad_sequence(sequences=audios, batch_first=True, padding_value=0.0)
    result_batch["audio"] = padded_audio
    result_batch["labels"] = torch.tensor(labels)

    return result_batch
