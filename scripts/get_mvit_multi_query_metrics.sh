# The script generates the class-agnostic detections for 'MDef-DETR' model for different queries and then combines the detections from each query.
# The arguments of this script are as follows,
# 1st Argument: path to directory containing the 8 evaluation datasets (Pascal VOC, COCO, Clipart, Comic, Kitchen, KITTI and DOTA)
# 2nd Argument: model checkpoints path

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"




DATASET_BASE_DIR=${1:-$PROJECT_ROOT/data}
CHECKPOINTS_PATH=${2:-$PROJECT_ROOT/checkpoints/MDef_DETR_r101_epoch20.pth}

echo DATASET_BASE_DIR $DATASET_BASE_DIR
echo CHECKPOINTS_PATH $CHECKPOINTS_PATH

MODEL_NAME=mdef_detr
# For Pascal VOC, COCO, Clipart, Comic and Watercolor datasets
TEXT_QUERIES_SET_1='[all objects,all entities,all visible entities and objects,all obscure entities and objects]'
# For Kitchen, KITTI and DOTA datasets
TEXT_QUERIES_SET_2='[all objects,all entities,all visible entities and objects,all obscure entities and objects,all small objects]'
export PYTHONPATH="./:$PYTHONPATH"

# Pascal VOC
#echo "Running inference on Pascal VOC"
#python inference/main_mvit_multi_query.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/voc2007/JPEGImages" -c "$CHECKPOINTS_PATH" -tq_list "$TEXT_QUERIES_SET_1"
#echo "Combining detections"
#python utils/combine_detections.py -i "$DATASET_BASE_DIR/voc2007/$MODEL_NAME"

# COCO
echo "Running inference on COCO"
python inference/main_mvit_multi_query.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/hypersim-coco/images" -c "$CHECKPOINTS_PATH" -tq_list "$TEXT_QUERIES_SET_1"
echo "Combining detections"
python utils/combine_detections.py -i "$DATASET_BASE_DIR/hypersim-coco/$MODEL_NAME"

# KITTI
#echo "Running inference on KITTI"
#python inference/main_mvit_multi_query.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/kitti/JPEGImages" -c "$CHECKPOINTS_PATH" -tq_list "$TEXT_QUERIES_SET_2"
#echo "Combining detections"
#python utils/combine_detections.py -i "$DATASET_BASE_DIR/kitti/$MODEL_NAME"

# Compute Class-agnostic object detection metrics
echo "Evaluating"
python evaluation/class_agnostic_od/get_multi_dataset_eval_metrics.py -m "$MODEL_NAME"
