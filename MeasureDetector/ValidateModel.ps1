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

echo "Validate with $($configuration) configuration"

Start-Transcript -path "$($pathToTranscript)/Validate-$($configuration).txt" -append
python research/object_detection/legacy/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToData)/$($configuration)" --eval_dir="$($pathToData)/$($configuration)/eval"
Stop-Transcript
