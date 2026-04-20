import numpy as np
import matplotlib, torch, os, glob
import pandas as pd
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from . import plot as myplot

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

class Sam3Mask:
    '''Class to run SAM3 on multiple images and prompts, and convert results to COCO format.'''
    def __init__(self,
                 model_name="facebook/sam3",
                 processor_name="facebook/sam3",
                 ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam3Processor.from_pretrained(processor_name)

    def run_multi_image_multi_prompt(
        self,
        images,
        prompts_per_image=None,
        input_boxes_per_image=None,
        input_boxes_labels_per_image=None,
        batch_size=8,
        threshold=0.5,
        mask_threshold=0.5,
    ):
        '''Run SAM3 on multiple images with multiple prompts, and return results grouped by image.
        Inputs:
        - images: list of PIL Images
        - prompts_per_image: list of text prompts (strings) or list of lists of text prompts (one list per image)
        - input_boxes_per_image: list of lists of boxes (one list per image), where each box is [x1, y1, x2, y2]
        - input_boxes_labels_per_image: list of lists of labels (one list per image), where each label corresponds to a box
        - batch_size: number of prompts to process in one batch
        - threshold: confidence threshold for mask prediction
        - mask_threshold: threshold for binarizing masks
        Output:
        - grouped_results: dict mapping image index to list of {"prompt": prompt, "result": result} dicts, where result contains "masks", "scores", etc. as returned by the processor's post-processing function.
        '''
        
        n_images = len(images)
        grouped_results = {i: [] for i in range(n_images)}

        # Send home early if no images to process
        if n_images == 0:   
            print("Warning: no images provided. Returning empty results.")
            return grouped_results

        # Normalize text prompts:
        # 1) prompts_per_image = ["sign", "support pole"]
        # 2) prompts_per_image = [["sign", "support pole"]]
        prompts = []
        if prompts_per_image:
            if isinstance(prompts_per_image[0], list):
                if len(prompts_per_image) > 1:
                    print("Warning: multiple prompt lists provided. Using only the first list for all images.")
                prompts = prompts_per_image[0]
            else:
                prompts = prompts_per_image

        prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
        has_text_prompts = len(prompts) > 0

        # Normalize bbox prompts to per-image lists of [x1, y1, x2, y2]
        boxes_per_image = [[] for _ in range(n_images)]
        if input_boxes_per_image is not None:
            if len(input_boxes_per_image) != n_images:
                raise ValueError("input_boxes_per_image must have one entry per image.")

            for image_idx, image_boxes in enumerate(input_boxes_per_image):
                if image_boxes is None:
                    continue

                if image_boxes and isinstance(image_boxes[0], (int, float)):
                    image_boxes = [image_boxes]

                normalized_boxes = []
                for box in image_boxes:
                    if box is None:
                        continue
                    if len(box) != 4:
                        raise ValueError(f"Each box must be [x1, y1, x2, y2]. Invalid box for image {image_idx}: {box}")
                    normalized_boxes.append([float(v) for v in box])

                boxes_per_image[image_idx] = normalized_boxes

        has_box_prompts = any(len(image_boxes) > 0 for image_boxes in boxes_per_image)

        # Normalize labels; every box must have one matching label.
        if has_box_prompts:
            if input_boxes_labels_per_image is None:
                labels_per_image = [[1] * len(image_boxes) for image_boxes in boxes_per_image]
            else:
                if len(input_boxes_labels_per_image) != n_images:
                    raise ValueError("input_boxes_labels_per_image must have one entry per image.")

                labels_per_image = [[] for _ in range(n_images)]
                for image_idx, image_labels in enumerate(input_boxes_labels_per_image):
                    n_boxes = len(boxes_per_image[image_idx])

                    if n_boxes == 0:
                        labels_per_image[image_idx] = []
                        continue

                    if image_labels is None:
                        labels_per_image[image_idx] = [1] * n_boxes
                        continue

                    if len(image_labels) != n_boxes:
                        raise ValueError(
                            f"input_boxes_labels_per_image[{image_idx}] has {len(image_labels)} labels, "
                            f"but input_boxes_per_image[{image_idx}] has {n_boxes} boxes."
                        )

                    labels_per_image[image_idx] = [int(label) for label in image_labels]
        else:
            labels_per_image = [[] for _ in range(n_images)]

        if not has_text_prompts and not has_box_prompts:
            print("Warning: no valid text prompts or bbox prompts found. Returning empty results.")
            return grouped_results

        if has_text_prompts and has_box_prompts:
            mode = "text_and_bbox"
        elif has_text_prompts:
            mode = "text_only"
        else:
            mode = "bbox_only"

        # One task = one inference row in the processor batch.
        # Task tuple: (image_idx, image, text_prompt, box_xyxy, box_label, output_prompt_name)
        tasks = []
        for image_idx, img in enumerate(images):
            if mode == "text_only":
                for prompt in prompts:
                    tasks.append((image_idx, img, prompt, None, None, prompt))
            elif mode == "bbox_only":
                for box_idx, (box, box_label) in enumerate(zip(boxes_per_image[image_idx], labels_per_image[image_idx])):
                    tasks.append((image_idx, img, None, box, box_label, f"bbox_{box_idx}"))
            else:
                for box, box_label in zip(boxes_per_image[image_idx], labels_per_image[image_idx]):
                    for prompt in prompts:
                        tasks.append((image_idx, img, prompt, box, box_label, prompt))

        for start in range(0, len(tasks), batch_size):
            chunk = tasks[start : start + batch_size]
            chunk_image_ids = [item[0] for item in chunk]
            chunk_images = [item[1] for item in chunk]
            chunk_prompts = [item[2] for item in chunk]
            chunk_boxes = [item[3] for item in chunk]
            chunk_box_labels = [item[4] for item in chunk]
            chunk_output_prompts = [item[5] for item in chunk]

            # print(f"Processing batch {start // batch_size + 1} on images {chunk_image_ids[0]} to {chunk_image_ids[-1]}")
            processor_kwargs = {
                "images": chunk_images,
                "return_tensors": "pt",
            }

            if mode in ("text_only", "text_and_bbox"):
                processor_kwargs["text"] = chunk_prompts

            if mode in ("bbox_only", "text_and_bbox"):
                processor_kwargs["input_boxes"] = [[box] for box in chunk_boxes]
                processor_kwargs["input_boxes_labels"] = [[label] for label in chunk_box_labels]

            inputs = self.processor(**processor_kwargs).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            for image_idx, prompt, result in zip(chunk_image_ids, chunk_output_prompts, batch_results):
                grouped_results[image_idx].append({"prompt": prompt, "result": result})

        return grouped_results

    def build_plot_df_from_prompt_items(
            self,
            prompt_items, 
            prompts,
            mode=None):
        
        '''Convert SAM3 processor results to a DataFrame for plotting and COCO conversion.
        Inputs:
        - prompt_items: list of {"prompt": prompt, "result": result} dicts, where result contains "masks", "scores", etc. as returned by the processor's post-processing function.
        - prompts: list of original prompts corresponding to the results, used to derive category names. For "bbox_only" mode, these are the actual category names. For other modes, these are the text prompts.
        - mode: "text_only", "bbox_only", or "text_and_bbox", which determines how to derive category names from prompts and prompt_items.
        
        Output:
        - df_plot: DataFrame with columns ["category_id", "category_name", "segmentation", "bbox", "area", "score"] containing the results for plotting and COCO conversion.
        - category: DataFrame with columns ["category_id", "name"] mapping category IDs to category names.
        '''

        # For bbox_only mode, use only the input prompts (actual category text).
        # For other modes, derive categories from both prompts and prompt_items.
        if mode == "bbox_only":
            prompt_names = [p for p in prompts if isinstance(p, str) and p.strip()]
        else:
            prompt_names = [p for p in prompts if isinstance(p, str) and p.strip()]
            prompt_names.extend(
                [
                    item.get("prompt")
                    for item in prompt_items
                    if isinstance(item.get("prompt"), str) and item.get("prompt").strip()
                ]
            )
        prompt_names = list(dict.fromkeys(prompt_names))
        
        category = pd.DataFrame({
            "category_id": list(range(1, len(prompt_names) + 1)),
            "name": prompt_names,
            })
        prompt_to_id = {p: i + 1 for i, p in enumerate(prompt_names)}

        rows = []
        for item in prompt_items:
            prompt = item["prompt"]
            result = item["result"]
            masks = result.get("masks")
            scores = result.get("scores")
            
            if masks is None:
                continue
            
            # Process ALL masks in this prompt result
            for m_idx in range(len(masks)):
                mask_tensor = masks[m_idx]
                if hasattr(mask_tensor, "detach"):
                    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
                else:
                    mask_np = np.array(mask_tensor, dtype=np.uint8)
            
                polygons, area, _ = myplot.mask_to_polygon(mask_np, multiply=True)
                if polygons is None or area == 0:
                    continue
            
                bbox = myplot.find_remasked_bbox(mask_np)
                score = 0.0
                if scores is not None and len(scores) > m_idx:
                    score_val = scores[m_idx]
                    if hasattr(score_val, "detach"):
                        score = float(score_val.detach().cpu().item())
                    else:
                        score = float(score_val)
            
                # For bbox_only mode, use the actual prompt (category text), not synthetic bbox_{i}
                category_name = prompts[0] if mode == "bbox_only" else prompt
                
                rows.append({
                    "category_id": prompt_to_id.get(category_name, 1),
                    "category": category_name,  # Add category name for COCO conversion
                    "segmentation": polygons,
                    "bbox": bbox,
                    "area": float(area),
                    "score": score,
                })
            
        df_plot = pd.DataFrame(rows, columns=["category_id", "category", "segmentation", "bbox", "area", "score"])
        return df_plot, category