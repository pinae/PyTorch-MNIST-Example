from typing import Optional, Callable, Tuple, Any
from torchvision.datasets import MNIST
from PIL import Image
from os import path
import numpy as np


class Confusing_MNIST(MNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            match_ratio=0.5
    ) -> None:
        super().__init__(root=root,
                         train=train,
                         transform=None,
                         target_transform=None,
                         download=download)
        self.second_transform = transform
        self.second_target_transform = target_transform
        self.confusion_vectors = (np.random.random(10 * 28) * 256).astype(np.uint8).reshape(10, 28)
        self.match_ratio = match_ratio

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index=index)
        img_data = np.array(img)
        if np.random.random() < self.match_ratio:
            confusion_vector = self.confusion_vectors[target]
        else:
            candidates = list(range(10))
            candidates.pop(target)
            confusion_vector = self.confusion_vectors[np.random.choice(candidates)]
        img_data = np.insert(img_data, 28, values=confusion_vector, axis=-2)
        img = Image.fromarray(img_data, mode="L")

        if self.second_transform is not None:
            img = self.second_transform(img)

        if self.second_target_transform is not None:
            target = self.second_target_transform(target)

        return img, target

    @property
    def raw_folder(self) -> str:
        return path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return path.join(self.root, "MNIST", "processed")


if __name__ == "__main__":
    confusing_mnist = Confusing_MNIST(path.join('..', 'data'))
    img, target = confusing_mnist[0]
    print(np.array(img).shape, np.array(img).dtype)
    print(target)
    img.save('image0.png')
    img, target = MNIST(path.join('..', 'data'))[0]
    print(np.array(img).shape, np.array(img).dtype)
    print(target)
