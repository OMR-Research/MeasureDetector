# Stave Detection
This part of the repository contains a similar training procedure as the MeasureDetector,
but focuses on detecting staves, especially using Mask R-CNN, not just Fast R-CNN.

To perform the training on the MUSCIMA++ dataset, run the following:

```bash
python prepare_muscima-pp_dataset.py
python create_tf_record_from_individual_json_files.py
python research/object_detection/legacy/train.py --pipeline_config_path="CONFIGURATION_PATH.config" --train_dir="PATH_TO_DATA"
```

```bash
python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="configurations/mask_rcnn_inception_resnet_v2_atrous_muscima-pp_staves.config" --model_dir="data/mask_rcnn_inception_resnet_v2_atrous_muscima-pp_staves"
```
For example paths, see [TrainModel.ps1](TrainModel.ps1).
 