import os
import cv2


class ReadImages:
    def __init__(self):
        self.images: list = []
        self.names: list = []

    def read_images(self, database_path: str):
        data = os.listdir(database_path)

        for lis in data:
            image_path = os.path.join(database_path, lis)
            imgdb = cv2.imread(image_path)

            if imgdb is not None:
                self.images.append(imgdb)
                self.names.append(os.path.splitext(lis)[0])
            else:
                print(f"Warning: Failed to load image {image_path}. Skipping...")

        return self.images, self.names
