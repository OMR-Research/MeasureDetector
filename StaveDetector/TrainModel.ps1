$pathToGitRoot = "C:/Users/Alex/Repositories/MeasureDetector"
$pathToSourceRoot = "$($pathToGitRoot)/StaveDetector"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
$pathToConfigurations = "$($pathToSourceRoot)/configurations"
$pathToData = "$($pathToSourceRoot)/data"
cd $pathToGitRoot

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
$configuration = "faster_rcnn_inception_resnet_v2_atrous_fine_grid_staves"
$configuration = "mask_rcnn_inception_resnet_v2_atrous_muscima-pp_staves"
$configuration = "mask_rcnn_resnet50_muscima-pp_staves"

echo "Training with $($configuration) configuration"

# Legacy slim-based
Start-Transcript -path "$($pathToTranscript)/Train-$($configuration).txt" -append
python research/object_detection/legacy/train.py --pipeline_config_path="$($pathToConfigurations)/$($configuration).config" --train_dir="$($pathToData)/$($configuration)"
Stop-Transcript

#Start-Transcript -path "$($pathToTranscript)/Validate-$($configuration).txt" -append
#python research/object_detection/legacy/eval.py --logtostderr --pipeline_config_path="$($pathToConfigurations)/$($configuration).config" --checkpoint_dir="$($pathToData)/$($configuration)" --eval_dir="$($pathToData)/$($configuration)/eval"
#Stop-Transcript

# # Estimator-based
# Start-Transcript -path "$($pathToTranscript)/TrainEval-$($configuration).txt" -append
# python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/checkpoints-$($configuration)"
# Stop-Transcript
