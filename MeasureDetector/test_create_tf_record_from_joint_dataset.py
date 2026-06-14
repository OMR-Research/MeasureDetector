import importlib
import sys
import types
import unittest

# The module under test imports tensorflow, contextlib2, and other heavy deps
# at module level. We stub them out so we can import just the logic we need.
tf_stub = types.ModuleType("tensorflow")
tf_stub.train = types.SimpleNamespace(Example=object)
tf_stub.gfile = types.SimpleNamespace(GFile=object)
tf_stub.Session = object
tf_stub.Graph = object
tf_stub.GraphDef = object
tf_stub.import_graph_def = lambda *a, **kw: None
tf_stub.get_default_graph = lambda: None
sys.modules["tensorflow"] = tf_stub

for stub in ["contextlib2",
             "object_detection", "object_detection.dataset_tools",
             "object_detection.dataset_tools.tf_record_creation_util",
             "object_detection.utils", "object_detection.utils.dataset_util",
             "object_detection.utils.label_map_util"]:
    sys.modules.setdefault(stub, types.ModuleType(stub))

tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.tqdm = lambda *a, **kw: iter([])
sys.modules["tqdm"] = tqdm_stub

errors_stub = types.ModuleType("MeasureDetector.errors")
errors_stub.InvalidImageFormatError = Exception
errors_stub.InvalidImageError = Exception
errors_stub.InvalidAnnotationError = Exception
sys.modules["MeasureDetector.errors"] = errors_stub

# PIL.Image must expose an `Image` attribute for the `from PIL.Image import Image` line
pil_image_stub = types.ModuleType("PIL.Image")
pil_image_stub.Image = object
sys.modules.setdefault("PIL", types.ModuleType("PIL"))
sys.modules["PIL.Image"] = pil_image_stub

# Stub encode_sample_into_tensorflow_sample used at module level
individual_json_stub = types.ModuleType("MeasureDetector.create_tf_record_from_individual_json_files")
individual_json_stub.encode_sample_into_tensorflow_sample = lambda *a, **kw: None
sys.modules["MeasureDetector.create_tf_record_from_individual_json_files"] = individual_json_stub

from MeasureDetector.create_tf_record_from_joint_dataset import attempt_to_find_sample_that_is_not_yet_in_dataset


class TestAttemptToFindSampleThatIsNotYetInDataset(unittest.TestCase):

    def test_returns_none_for_empty_category_with_reuse_allowed(self):
        result = attempt_to_find_sample_that_is_not_yet_in_dataset(
            all_items_in_category=[],
            already_used_sample_paths=[],
            samples_to_ignore=[],
            allow_sample_reuse=True,
        )
        self.assertIsNone(result)

    def test_returns_none_for_empty_category_with_reuse_forbidden(self):
        result = attempt_to_find_sample_that_is_not_yet_in_dataset(
            all_items_in_category=[],
            already_used_sample_paths=[],
            samples_to_ignore=[],
            allow_sample_reuse=False,
        )
        self.assertIsNone(result)

    def test_returns_none_when_all_samples_used_and_reuse_forbidden(self):
        samples = [{'path': 'a.png'}, {'path': 'b.png'}]
        result = attempt_to_find_sample_that_is_not_yet_in_dataset(
            all_items_in_category=samples,
            already_used_sample_paths=['a.png', 'b.png'],
            samples_to_ignore=[],
            allow_sample_reuse=False,
        )
        self.assertIsNone(result)

    def test_returns_sample_when_all_used_and_reuse_allowed(self):
        samples = [{'path': 'a.png'}, {'path': 'b.png'}]
        result = attempt_to_find_sample_that_is_not_yet_in_dataset(
            all_items_in_category=samples,
            already_used_sample_paths=['a.png', 'b.png'],
            samples_to_ignore=[],
            allow_sample_reuse=True,
        )
        self.assertIn(result, samples)

    def test_returns_fresh_sample_when_available(self):
        samples = [{'path': 'a.png'}, {'path': 'b.png'}]
        result = attempt_to_find_sample_that_is_not_yet_in_dataset(
            all_items_in_category=samples,
            already_used_sample_paths=['a.png'],
            samples_to_ignore=[],
            allow_sample_reuse=False,
        )
        self.assertEqual(result, {'path': 'b.png'})

    def test_ignores_samples_in_error_list(self):
        samples = [{'path': 'a.png'}, {'path': 'b.png'}]
        result = attempt_to_find_sample_that_is_not_yet_in_dataset(
            all_items_in_category=samples,
            already_used_sample_paths=[],
            samples_to_ignore=['a.png'],
            allow_sample_reuse=False,
        )
        self.assertEqual(result, {'path': 'b.png'})


if __name__ == '__main__':
    unittest.main()
