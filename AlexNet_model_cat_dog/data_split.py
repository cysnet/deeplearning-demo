ROOT_TRAIN = r"data/dogs-vs-cats-redux-kernels-edition/test/test"

import os
import shutil


def split_data():
    cat_dir = os.path.join(ROOT_TRAIN, "cat")
    dog_dir = os.path.join(ROOT_TRAIN, "dog")

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    for filename in os.listdir(ROOT_TRAIN):
        if filename.startswith("cat") and filename.endswith(".jpg"):
            shutil.move(
                os.path.join(ROOT_TRAIN, filename), os.path.join(cat_dir, filename)
            )
        elif filename.startswith("dog") and filename.endswith(".jpg"):
            shutil.move(
                os.path.join(ROOT_TRAIN, filename), os.path.join(dog_dir, filename)
            )


split_data()
