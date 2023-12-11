#%% 

import pickle
from pathlib import Path
import json
import numpy as np
# %%

DATA_PATH=Path(__file__).parent.parent.parent / "data" / "hypersim-coco" / "mdef_detr" / "combined.pkl"
ANNO_PATH=Path(__file__).parent.parent.parent / "data" / "hypersim-coco" / "instances_val2017.json"
OUT_PATH=Path(__file__).parent.parent.parent / "data" / "hypersim-coco" / "mdef_detr" / "hypersim_coco_detections.json"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

annotations = json.load(ANNO_PATH.open("r"))

# %%
images_no_ext = data.keys()
image_to_id = { i["file_name"].split(".")[0]: i["id"] for i in annotations["images"]}

detections=[]
for image in images_no_ext:
    id = image_to_id[image]
    category_id = 1
    boxes, scores = data[image]
    # take top 50 after sorting by scores
    top_50_indicies = np.argsort(scores)[-50:]
    boxes = np.array(boxes)[top_50_indicies, :].tolist()
    scores = np.array(scores)[top_50_indicies].tolist()
    for box, score in zip(boxes, scores):
        # append detection in coco format
        detections.append({
            "image_id": id,
            "category_id": category_id,
            "bbox": box,
            "score": score
        })
        






# %%
json.dump(detections, OUT_PATH.open("w+"), indent=2)
# %%
