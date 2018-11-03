$pathToGitRoot = "C:/Users/Alex/Repositories/MeasureDetector"
$pathToSourceRoot = "$($pathToGitRoot)/MeasureDetector"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
$pathToData = "$($pathToSourceRoot)/data"
cd $pathToGitRoot

#echo "Appending required paths to temporary PYTHONPATH"
#$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToGitRoot)/research/slim;$($pathToSourceRoot)"

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
$configuration = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_edirom"
$configuration = "faster_rcnn_inception_v2_edirom"

echo "Validate with $($configuration) configuration"

Start-Transcript -path "$($pathToTranscript)/Validate-$($configuration).txt" -append
python research/object_detection/legacy/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToData)/checkpoints-$($configuration)-train" --eval_dir="$($pathToData)/checkpoints-$($configuration)-validate"
Stop-Transcript
