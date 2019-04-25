$pathToGitRoot = "C:/Users/Alex/Repositories/MeasureDetector"
$pathToSourceRoot = "$($pathToGitRoot)/MeasureDetector"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
$pathToData = "$($pathToSourceRoot)/data"
cd $pathToGitRoot

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
$configuration = "faster_rcnn_inception_resnet_v2_atrous_all_datasets"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_all_datasets_fine_grid"
# $configuration = "faster_rcnn_inception_v2_all_datasets"
# $configuration = "faster_rcnn_resnet50_all_datasets"
# $configuration = "faster_rcnn_resnet50_all_datasets_fine_grid"
# $configuration = "faster_rcnn_resnet50_all_datasets_high_res"
# $configuration = "faster_rcnn_resnet50_all_datasets_high_res_no_pretrain"
# $configuration = "faster_rcnn_resnet101_all_datasets_fine_grid"

echo "Training with $($configuration) configuration"

# Legacy slim-based
Start-Transcript -path "$($pathToTranscript)/Train-$($configuration).txt" -append
python research/object_detection/legacy/train.py --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToData)/$($configuration)"
Stop-Transcript

# # Estimator-based
# Start-Transcript -path "$($pathToTranscript)/TrainEval-$($configuration).txt" -append
# python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/checkpoints-$($configuration)"
# Stop-Transcript
