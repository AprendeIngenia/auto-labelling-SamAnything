from tqdm import tqdm
import os
import cv2
import numpy as np
import torch
import supervision as sv
from typing import Any
from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from database.read_database import ReadImages


class AutoLabellingSaveData:
    def __init__(self):
        self.data = ReadImages()

        # variables
        self.box_threshold: float = 0.35
        self.text_threshold: float = 0.25
        self.min_image_area: float = 0.002
        self.max_image_area: float = 0.80
        self.approx: float = 0.75

        self.input_path: str = 'database/images/'
        self.out_path: str = 'database/annotations/'
        self.home: str = os.getcwd()

        self.save: bool = False
        self.draw: bool = True
        self.mask_generator_flag: bool = True

        self.extensions: list = ['jpg', 'jpeg', 'png']
        self.classes: list = ['beer glass']
        self.results_masks: list = []

        self.images: dict = {}
        self.annotations: dict = {}

        self.sam_predictor: Any = None
        self.mask_generator: Any = None
        self.sam_model: Any = None
        self.device: Any = None

    def save_data(self, images: dict, annotations: dict, data_format: str):
        if data_format == 'voc':
            sv.Dataset(
                classes=self.classes,
                images=images,
                annotations=annotations
            ).as_pascal_voc(
                annotations_directory_path=self.out_path,
                min_image_area_percentage=self.min_image_area,
                max_image_area_percentage=self.max_image_area,
                approximation_percentage=self.approx
            )

    def config_grounding_model(self) -> Any:
        config_path = os.path.join(self.home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        check_point_path = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'
        model = Model(config_path, check_point_path, device="cuda")
        return model

    def config_sam_anything(self) -> Any:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        check_point_path = self.home + '/weights_sam/sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        model_sam = sam_model_registry[model_type](check_point_path).to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(model_sam)
        sam_predictor = SamPredictor(model_sam)
        return model_sam, mask_generator, sam_predictor

    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        self.results_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            self.results_masks.append(masks[index])
        return np.array(self.results_masks)

    def main(self):
        grounding_model = self.config_grounding_model()
        self.sam_model, self.mask_generator, self.sam_predictor = self.config_sam_anything()

        # read images
        images_path = sv.list_files_with_extensions(directory=self.input_path, extensions=self.extensions)

        for image_path in tqdm(images_path):
            image_name = image_path.name
            image = cv2.imread(image_path)

            detections = grounding_model.predict_with_classes(
                image=image,
                classes=self.classes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            detections = detections[detections.class_id != None]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections.mask = self.segment(
                sam_predictor=self.sam_predictor,
                image=image_rgb,
                xyxy=detections.xyxy
            )
            self.images[image_name] = image
            self.annotations[image_name] = detections

        self.save_data(self.images, self.annotations, data_format='voc')


auto_labeling = AutoLabellingSaveData()
auto_labeling.main()
