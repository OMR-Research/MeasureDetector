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

echo "Training with $($configuration) configuration"

# Legacy slim-based
Start-Transcript -path "$($pathToTranscript)/Train-$($configuration).txt" -append
python research/object_detection/legacy/train.py --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToData)/checkpoints-$($configuration)-train"
Stop-Transcript

# # Estimator-based
# Start-Transcript -path "$($pathToTranscript)/TrainEval-$($configuration).txt" -append
# python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/$($configuration)"
# Stop-Transcript


# C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\training_scripts\ValidateModel.ps1