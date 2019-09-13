from omrdatasettools.image_generators.MeasureVisualizer import MeasureVisualizer

visualizer = MeasureVisualizer(False, False, True)
visualizer.draw_bounding_boxes_for_all_images_in_directory(r"E:\Stave Detection\deep_scores_temp",
                                          r"E:\Stave Detection\deep_scores_temp\annotations")