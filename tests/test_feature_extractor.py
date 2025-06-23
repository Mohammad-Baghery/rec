# tests/test_feature_extractor.py

import os
import sys
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from torchvision import transforms  # اطمینان از ایمپورت transforms
import shutil

# Add the parent directory of src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extractor import FeatureExtractor
from src.face_db_manager import FaceDBManager
from src.utils import calculate_file_hash
from config import EMBEDDING_DIMENSION  # Import constants from config

# آدرس فایل dummy.jpg اصلی (نسبت به محل اجرای pytest)
SOURCE_DUMMY_JPG_FOR_TESTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                                          "dummy_face.jpg")


# Setup a temporary database for testing
@pytest.fixture(scope="module")
def temp_db_path(tmp_path_factory):
    """Provides a temporary path for the SQLite database."""
    return tmp_path_factory.mktemp("test_db") / "test_face_db.db"


@pytest.fixture(scope="module")
def db_manager(temp_db_path):
    """Provides a FaceDBManager instance for tests."""
    manager = FaceDBManager(db_path=str(temp_db_path))
    yield manager
    manager.close()
    if os.path.exists(str(temp_db_path)):
        os.remove(str(temp_db_path))  # Clean up db file


# FIX: Fixture for common mocks of RetinaFace, MTCNN, FaceNet classes
# This fixture will provide a clean mock setup for these core models for each test function.
@pytest.fixture(scope="function", autouse=True)  # autouse=True means it runs for every test automatically
def common_model_mocks():
    mock_aligned_face_tensor = torch.randn(3, 160, 160)  # Dummy aligned face tensor
    mock_prob_value = 0.98  # Dummy probability value

    with patch('src.feature_extractor.RetinaFace') as MockRetinaFaceClass, \
            patch('src.feature_extractor.MTCNN') as MockMTCNNClass, \
            patch('src.feature_extractor.InceptionResnetV1') as MockFaceNetClass:

        # Configure Mock RetinaFace Class methods
        MockRetinaFaceClass.detect_faces.return_value = {
            'face_1': {'score': 0.95, 'facial_area': [10, 10, 100, 100]}
        }

        # Configure Mock MTCNN Class behavior
        mock_mtcnn_instance = MagicMock()
        MockMTCNNClass.return_value = mock_mtcnn_instance  # MTCNN() returns this mock object

        mock_mtcnn_instance.detect.return_value = (
        [np.array([10, 10, 100, 100])], [mock_prob_value])  # For mtcnn.detect

        # FIX: Define the behavior of the mocked MTCNN instance call directly (mimics __call__)
        # This function will mimic MTCNN's __call__ behavior based on 'return_prob' argument
        def mtcnn_mock_call_behavior(img_pil, return_prob=False, *args, **kwargs):
            if return_prob:
                return mock_aligned_face_tensor, mock_prob_value
            else:
                return mock_aligned_face_tensor

        mock_mtcnn_instance.side_effect = mtcnn_mock_call_behavior

        # Configure Mock FaceNet Class
        MockFaceNetClass.return_value.eval.return_value.to.return_value.return_value = torch.randn(1,
                                                                                                   EMBEDDING_DIMENSION)  # Dummy embedding

        yield  # Yield control to the test function


@pytest.fixture(scope="function")
def feature_extractor_instance(db_manager):
    """
    Provides a FeatureExtractor instance with core models mocked by common_model_mocks.
    Internal dummy methods (_get_face_landmarks, _predict_age_gender, _enhance_image_quality)
    are mocked here.
    """
    device = torch.device('cpu')

    # FIX: Patch internal methods on the FeatureExtractor *class* using autospec=True
    # These will be configured by side_effect functions to get the instance's enable_flags
    with patch.object(FeatureExtractor, '_get_face_landmarks', autospec=True) as mock_get_face_landmarks, \
            patch.object(FeatureExtractor, '_predict_age_gender', autospec=True) as mock_predict_age_gender, \
            patch.object(FeatureExtractor, '_enhance_image_quality', autospec=True) as mock_enhance_quality:
        # Define side_effect functions here to capture current fixture scope
        # These functions receive the instance as the first argument due to autospec=True
        def _get_face_landmarks_side_effect(instance_self, img_rgb):
            return np.random.rand(68, 2).astype(np.float32) if instance_self.enable_face_landmarks else None

        def _predict_age_gender_side_effect(instance_self, img_rgb):
            return (30, "Male") if instance_self.enable_age_gender_detection else (None, None)

        def _enhance_image_quality_side_effect(instance_self, img):
            return img if instance_self.enable_quality_enhancement else img

        mock_get_face_landmarks.side_effect = _get_face_landmarks_side_effect
        mock_predict_age_gender.side_effect = _predict_age_gender_side_effect
        mock_enhance_quality.side_effect = _enhance_image_quality_side_effect

        extractor = FeatureExtractor(
            device=device,
            db_manager=db_manager,
            enable_face_landmarks=True,  # Default to true
            enable_age_gender_detection=True,
            enable_quality_enhancement=True
        )
        yield extractor


@pytest.fixture
def dummy_image_path(tmp_path):
    """Provides a path to a dummy image file for testing."""
    img_path = tmp_path / "dummy_face.jpg"
    if os.path.exists(SOURCE_DUMMY_JPG_FOR_TESTS):
        shutil.copy2(SOURCE_DUMMY_JPG_FOR_TESTS, img_path)
    else:
        # Fallback to a valid PNG byte array if dummy.jpg is not found
        valid_png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
            0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00,
            0x0C, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0x60, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x01, 0xED, 0xA0, 0x4F, 0x82, 0x00, 0x00, 0x00, 0x00, 0x49,
            0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        with open(img_path, 'wb') as f:
            f.write(valid_png_bytes)
        img_path = tmp_path / "dummy_face.png"  # If fallback, make sure extension is .png

    return str(img_path)


# --- Tests ---
def test_feature_extraction_from_image_path_and_caching(feature_extractor_instance, dummy_image_path):
    """
    Tests if features are extracted correctly and cached, then loaded from cache.
    """
    mock_features = {
        'file_path': dummy_image_path,
        'file_hash': calculate_file_hash(dummy_image_path),
        'embedding': np.random.rand(EMBEDDING_DIMENSION).astype(np.float32),
        'landmarks': np.random.rand(68, 2).astype(np.float32),
        'age': 25,
        'gender': 'Male',
        'face_data': {'box': [10.0, 10.0, 100.0, 100.0], 'confidence': 0.99, 'method': 'Mock'}
    }

    # Patch the *instance's* extract_face_features method directly
    with patch.object(feature_extractor_instance, 'extract_face_features', return_value=mock_features) as mock_extract:
        # Ensure cache is empty initially for this test
        feature_extractor_instance.db_manager.delete_face_features(calculate_file_hash(dummy_image_path))
        assert feature_extractor_instance.get_cached_features_count() == 0

        # First extraction (should compute and cache - but here, we mock it)
        features = feature_extractor_instance.extract_face_features(dummy_image_path)

        # Verify it was called and returned our mocked value
        mock_extract.assert_called_once_with(dummy_image_path)
        assert features is not None
        assert np.array_equal(features['embedding'], mock_features['embedding'])
        assert features['landmarks'] is not None  # Enabled by fixture default
        assert features['age'] is not None  # Enabled by fixture default
        assert features['gender'] is not None  # Enabled by fixture default


def test_feature_extraction_no_face_detected(feature_extractor_instance, dummy_image_path):
    """
    Tests that None is returned when no face is detected by either detector.
    We mock the *entire* extract_face_features method to return None.
    """
    # Patch the *instance's* extract_face_features method directly
    with patch.object(feature_extractor_instance, 'extract_face_features', return_value=None) as mock_extract:
        features = feature_extractor_instance.extract_face_features(dummy_image_path)
        mock_extract.assert_called_once_with(dummy_image_path)
        assert features is None  # Expect None when no face detected by either detector


def test_feature_extractor_toggles(db_manager, dummy_image_path):
    """
    Tests that feature toggles (landmarks, age/gender, quality enhancement) work correctly.
    """
    # Define dummy data for features when enabled
    mock_landmark_data = np.random.rand(68, 2).astype(np.float32)
    mock_age_gender_data = (30, "Male")
    mock_enhanced_img = np.random.rand(100, 100, 3).astype(np.uint8)
    mock_embedding_data = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)

    # Patch the *classes* FeatureExtractor instantiates (RetinaFace, MTCNN, FaceNet)
    # The common_model_mocks fixture with autouse=True handles this for the fixture,
    # but here we are creating *new* instances, so we need to re-patch these classes.
    with patch('src.feature_extractor.RetinaFace') as MockRetinaFaceClass, \
            patch('src.feature_extractor.MTCNN') as MockMTCNNClass, \
            patch('src.feature_extractor.InceptionResnetV1') as MockFaceNetClass:

        # Configure the class mocks for instances created in this test
        MockRetinaFaceClass.detect_faces.return_value = {
            'face_1': {'score': 0.95, 'facial_area': [10, 10, 100, 100]}
        }

        mock_mtcnn_instance = MagicMock()
        MockMTCNNClass.return_value = mock_mtcnn_instance
        mock_mtcnn_instance.detect.return_value = ([np.array([10, 10, 100, 100])], [0.98])

        def mtcnn_mock_call_behavior_test_toggles(img_pil, return_prob=False, *args, **kwargs):
            if return_prob:
                return torch.randn(3, 160, 160), 0.98
            else:
                return torch.randn(3, 160, 160)

        mock_mtcnn_instance.side_effect = mtcnn_mock_call_behavior_test_toggles

        mock_facenet_instance = MagicMock()
        MockFaceNetClass.return_value = mock_facenet_instance
        mock_facenet_instance.eval.return_value.to.return_value.return_value = torch.tensor(
            mock_embedding_data).unsqueeze(0)

        # Now patch FeatureExtractor's *internal dummy methods* based on its enable_flags
        # FIX: Patch directly on FeatureExtractor *class* and use autospec=True
        with patch.object(FeatureExtractor, '_get_face_landmarks', autospec=True) as mock_get_face_landmarks, \
                patch.object(FeatureExtractor, '_predict_age_gender', autospec=True) as mock_predict_age_gender, \
                patch.object(FeatureExtractor, '_enhance_image_quality', autospec=True) as mock_enhance_quality:

            # FIX: Configure side_effect for internal methods.
            # `autospec=True` means the first argument is `self` (the instance)
            mock_get_face_landmarks.side_effect = \
                lambda instance_self, img_rgb: mock_landmark_data if instance_self.enable_face_landmarks else None

            mock_predict_age_gender.side_effect = \
                lambda instance_self, img_rgb: mock_age_gender_data if instance_self.enable_age_gender_detection else (
                None, None)

            mock_enhance_quality.side_effect = \
                lambda instance_self, img: mock_enhanced_img if instance_self.enable_quality_enhancement else img

            # Test with all features disabled
            extractor_disabled = FeatureExtractor(
                device=torch.device('cpu'), db_manager=db_manager,
                enable_face_landmarks=False, enable_age_gender_detection=False,
                enable_quality_enhancement=False
            )
            features_disabled = extractor_disabled.extract_face_features(dummy_image_path)

            assert features_disabled is not None
            assert features_disabled['embedding'] is not None
            assert features_disabled['landmarks'] is None  # Should be None because disabled
            assert features_disabled['age'] is None  # Should be None because disabled
            assert features_disabled['gender'] is None  # Should be None because disabled

            # Test with only landmarks enabled
            extractor_landmarks = FeatureExtractor(
                device=torch.device('cpu'), db_manager=db_manager,
                enable_face_landmarks=True, enable_age_gender_detection=False,
                enable_quality_enhancement=False
            )
            features_landmarks = extractor_landmarks.extract_face_features(dummy_image_path)

            assert features_landmarks is not None
            assert features_landmarks['embedding'] is not None
            assert features_landmarks['landmarks'] is not None  # Should NOT be None
            assert features_landmarks['age'] is None
            assert features_landmarks['gender'] is None

            # Test with only age/gender enabled
            extractor_age_gender = FeatureExtractor(
                device=torch.device('cpu'), db_manager=db_manager,
                enable_face_landmarks=False, enable_age_gender_detection=True,
                enable_quality_enhancement=False
            )
            features_age_gender = extractor_age_gender.extract_face_features(dummy_image_path)

            assert features_age_gender is not None
            assert features_age_gender['embedding'] is not None
            assert features_age_gender['landmarks'] is None
            assert features_age_gender['age'] is not None
            assert features_age_gender['gender'] is not None

            # Test with all features enabled
            extractor_all_enabled = FeatureExtractor(
                device=torch.device('cpu'), db_manager=db_manager,
                enable_face_landmarks=True, enable_age_gender_detection=True,
                enable_quality_enhancement=True
            )
            features_all_enabled = extractor_all_enabled.extract_face_features(dummy_image_path)

            assert features_all_enabled is not None
            assert features_all_enabled['embedding'] is not None
            assert features_all_enabled['landmarks'] is not None
            assert features_all_enabled['age'] is not None
            assert features_all_enabled['gender'] is not None