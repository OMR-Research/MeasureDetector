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
$configuration = "mask_rcnn_resnet50_muscima-pp_staves"
$configuration = "mask_rcnn_inception_resnet_v2_atrous_muscima-pp_staves"

echo "Training with $($configuration) configuration"

# Estimator-based training
Start-Transcript -path "$($pathToTranscript)/TrainEval-$($configuration).txt" -append
python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/$($configuration)"
Stop-Transcript


Write-Host -NoNewLine 'Press any key to continue...';
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');