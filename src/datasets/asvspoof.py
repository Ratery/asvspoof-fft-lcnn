from pathlib import Path

from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    def __init__(self, protocol_path, audio_dir, *args, **kwargs):
        index = self._create_index(Path(protocol_path), Path(audio_dir))
        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path, audio_dir):
        index = []
        label_mapping = {"bonafide": 1, "spoof": 0}

        with protocol_path.open(mode="r", encoding="utf-8") as file:
            for line in tqdm(file.readlines()):
                speaker_id, audio_filename, system_id, _, key = line.strip().split()
                entry = {
                    "path": audio_dir / f"{audio_filename}.flac",
                    "label": label_mapping[key],
                    "metadata": {
                        "speaker_id": speaker_id,
                        "audio_filename": audio_filename,
                        "system_id": system_id,
                        "key": key,
                    },
                }
                index.append(entry)

        return index
