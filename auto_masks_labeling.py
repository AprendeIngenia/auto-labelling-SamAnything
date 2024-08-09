from time import time
import os
import cv2
import numpy as np
import torch
import supervision as sv
from PIL import Image
from typing import Any
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from database.read_database import ReadImages


class AutoLabellingInstanceSegmentation:
    def __init__(self):
        self.data = ReadImages()

        # variables
        self.cont: int = 0
        self.num_images: int = 0
        self.class_id: int = 0

        self.box_threshold: float = 0.38
        self.text_threshold: float = 0.25

        self.out_path: str = 'database/annotations/'
        self.prompt: str = 'id card'
        self.home: str = os.getcwd()

        self.save: bool = True
        self.draw: bool = False
        self.mask_generator_flag: bool = False

        self.images: list = []
        self.names: list = []
        self.bbox_info: list = []

        self.sam_predictor: Any = None
        self.mask_generator: Any = None
        self.sam_model: Any = None
        self.device: Any = None

    def save_data(self, image_copy: np.ndarray, list_info: list):
        timeNow = time()
        timeNow = str(timeNow)
        timeNow = timeNow.split('.')
        timeNow = timeNow[0] + timeNow[1]
        cv2.imwrite(f"{self.out_path}/{timeNow}.jpg", image_copy)
        for info in list_info:
            f = open(f"{self.out_path}/{timeNow}.txt", 'a')
            f.write(info)
            f.close()

    def config_grounding_model(self) -> Any:
        config_path = os.path.join(self.home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        check_point_path = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'
        model = load_model(config_path, check_point_path, device="cuda")
        return model

    def config_sam_anything(self) -> Any:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        check_point_path = self.home + '/weights_sam/sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        model_sam = sam_model_registry[model_type](check_point_path).to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(model_sam)
        sam_predictor = SamPredictor(model_sam)
        return model_sam, mask_generator, sam_predictor

    def total_mask_generator(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam_result = self.mask_generator.generate(image_rgb)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
        if self.draw:
            cv2.imshow("semantic segmentation", annotated_image)

    def show_mask(self, masks: Any, image: np.ndarray, random_color: bool = True, alpha: float = 0.5):
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        for mask in masks:
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
            else:
                color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

            mask = mask.cpu()
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
            annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)

        return np.array(annotated_frame_pil.convert("RGB"))

    def main(self):
        self.images, self.names = self.data.read_images('database/images')
        self.num_images = len(self.images)
        grounding_model = self.config_grounding_model()
        self.sam_model, self.mask_generator, self.sam_predictor = self.config_sam_anything()

        while self.cont < self.num_images:
            print('------------------------------------')
            print(f'name_image: {self.names[self.cont]}')

            process_image = self.images[self.cont]

            if process_image is None:
                print(f"Error loading image: {self.names[self.cont]}")
                self.cont += 1
                continue

            copy_image = process_image.copy()
            draw_image = process_image.copy()

            if self.mask_generator_flag:
                self.total_mask_generator(process_image)

            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )

            img_source = Image.fromarray(process_image).convert("RGB")
            img_transform, _ = transform(img_source, None)

            boxes, logits, phrases = predict(
                model=grounding_model,
                image=img_transform,
                caption=self.prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device="cuda"
            )

            h, w, _ = process_image.shape
            self.sam_predictor.set_image(image=process_image)
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
            transform_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, process_image.shape[:2]).to(
                self.device)

            masks, scores, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transform_boxes,
                multimask_output=True
            )
            if len(boxes) != 0:
                h, w, _ = process_image.shape
                xc, yc, an, al = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]

                # Error < 0
                if xc < 0: xc = 0
                if yc < 0: yc = 0
                if an < 0: an = 0
                if al < 0: al = 0
                # Error > 1
                if xc > 1: xc = 1
                if yc > 1: yc = 1
                if an > 1: an = 1
                if al > 1: al = 1

                self.bbox_info.append(f"{self.class_id} {xc} {yc} {an} {al}")
                x1, y1, x2, y2 = int(xc * w), int(yc * h), int(an * w), int(al * h)
                print(f"boxes: {boxes}\nxc: {x1} yc:{y1} w:{x2} h:{y2}")

                if self.save:
                    self.save_data(copy_image, self.bbox_info)

                if self.draw:
                    annotated_img = annotate(image_source=draw_image, boxes=boxes, logits=logits, phrases=phrases)
                    for mask_set in masks:
                        annotated_frame_with_mask = self.show_mask(mask_set, annotated_img, alpha=0.5)
                        out_frame = cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB)
                        cv2.imshow('Instance segmentation with SAM', out_frame)
                        cv2.waitKey(0)

            self.cont += 1


auto_labeling = AutoLabellingInstanceSegmentation()
auto_labeling.main()
