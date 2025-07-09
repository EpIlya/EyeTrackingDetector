import pytest
import cv2
from unittest.mock import Mock, patch, MagicMock
from main import GazeTracker, CONFIG


class TestGazeTracker:
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.gaze_tracker = GazeTracker()
    
    def test_initialization(self):
        """Тест инициализации GazeTracker"""
        assert self.gaze_tracker.camera is None
        assert self.gaze_tracker.calibrated is False
        assert self.gaze_tracker.horizontal_center == 0.5
        assert self.gaze_tracker.vertical_center == 0.5
        assert self.gaze_tracker.calibration_threshold == CONFIG["calibration_threshold"]
    
    @patch('cv2.VideoCapture')
    def test_initialize_camera(self, mock_video_capture):
        """Тест инициализации камеры"""
        mock_camera = Mock()
        mock_video_capture.return_value = mock_camera
        
        with patch.object(self.gaze_tracker, 'calibrate'):
            self.gaze_tracker.initialize_camera()
        
        mock_video_capture.assert_called_once_with(0)
        assert self.gaze_tracker.camera == mock_camera
    
    @patch('cv2.VideoCapture')
    def test_detect_gaze_without_camera(self, mock_video_capture):
        """Тест обнаружения взгляда без инициализированной камеры"""
        with pytest.raises(ValueError, match="Камера не инициализирована"):
            self.gaze_tracker.detect_gaze()
    
    @patch('cv2.VideoCapture')
    @patch('cv2.resize')
    @patch('cv2.putText')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_detect_gaze_success(self, mock_wait_key, mock_imshow, mock_put_text, mock_resize, mock_video_capture):
        """Тест успешного обнаружения взгляда"""
        mock_camera = Mock()
        mock_video_capture.return_value = mock_camera
        mock_camera.read.return_value = (True, Mock())
        
        mock_resize.return_value = Mock()
        mock_wait_key.return_value = -1
        
        mock_gaze = Mock()
        mock_gaze.refresh.return_value = None
        mock_annotated_frame = Mock()
        mock_annotated_frame.copy.return_value = Mock()
        mock_gaze.annotated_frame.return_value = mock_annotated_frame
        self.gaze_tracker.gaze = mock_gaze
        
        with patch.object(self.gaze_tracker, 'get_gaze_direction') as mock_get_direction:
            mock_get_direction.return_value = {"direction": "center"}
            
            self.gaze_tracker.calibrated = True
            self.gaze_tracker.camera = mock_camera
            
            result = self.gaze_tracker.detect_gaze()
            
            assert result == "center"
            mock_camera.read.assert_called_once()
            mock_gaze.refresh.assert_called_once()
    
    def test_get_gaze_direction_not_calibrated(self):
        """Тест получения направления взгляда без калибровки"""
        result = self.gaze_tracker.get_gaze_direction()
        assert result["direction"] == "not calibrated"
    
    def test_get_gaze_direction_blink(self):
        """Тест получения направления взгляда при моргании"""
        self.gaze_tracker.calibrated = True
        
        mock_gaze = Mock()
        mock_gaze.horizontal_ratio.return_value = None
        mock_gaze.vertical_ratio.return_value = None
        self.gaze_tracker.gaze = mock_gaze
        
        result = self.gaze_tracker.get_gaze_direction()
        assert result["direction"] == "blink"
    
    def test_get_gaze_direction_center(self):
        """Тест получения направления взгляда в центр"""
        self.gaze_tracker.calibrated = True
        self.gaze_tracker.horizontal_center = 0.5
        self.gaze_tracker.vertical_center = 0.5
        
        mock_gaze = Mock()
        mock_gaze.horizontal_ratio.return_value = 0.5
        mock_gaze.vertical_ratio.return_value = 0.5
        self.gaze_tracker.gaze = mock_gaze
        
        result = self.gaze_tracker.get_gaze_direction()
        assert result["direction"] == "center"
    
    def test_get_gaze_direction_left(self):
        """Тест получения направления взгляда влево"""
        self.gaze_tracker.calibrated = True
        self.gaze_tracker.horizontal_center = 0.5
        self.gaze_tracker.calibration_threshold = 0.1
        
        mock_gaze = Mock()
        mock_gaze.horizontal_ratio.return_value = 0.7
        mock_gaze.vertical_ratio.return_value = 0.5
        self.gaze_tracker.gaze = mock_gaze
        
        result = self.gaze_tracker.get_gaze_direction()
        assert result["direction"] == "left"
    
    def test_get_gaze_direction_right(self):
        """Тест получения направления взгляда вправо"""
        self.gaze_tracker.calibrated = True
        self.gaze_tracker.horizontal_center = 0.5
        self.gaze_tracker.calibration_threshold = 0.1
        
        mock_gaze = Mock()
        mock_gaze.horizontal_ratio.return_value = 0.3
        mock_gaze.vertical_ratio.return_value = 0.5
        self.gaze_tracker.gaze = mock_gaze
        
        result = self.gaze_tracker.get_gaze_direction()
        assert result["direction"] == "right"
    
    def test_get_gaze_direction_up(self):
        """Тест получения направления взгляда вверх"""
        self.gaze_tracker.calibrated = True
        self.gaze_tracker.vertical_center = 0.5
        self.gaze_tracker.calibration_threshold = 0.1
        
        mock_gaze = Mock()
        mock_gaze.horizontal_ratio.return_value = 0.5
        mock_gaze.vertical_ratio.return_value = 0.3
        self.gaze_tracker.gaze = mock_gaze
        
        result = self.gaze_tracker.get_gaze_direction()
        assert result["direction"] == "up"
    
    def test_get_gaze_direction_down(self):
        """Тест получения направления взгляда вниз"""
        self.gaze_tracker.calibrated = True
        self.gaze_tracker.vertical_center = 0.5
        self.gaze_tracker.calibration_threshold = 0.1
        
        mock_gaze = Mock()
        mock_gaze.horizontal_ratio.return_value = 0.5
        mock_gaze.vertical_ratio.return_value = 0.7
        self.gaze_tracker.gaze = mock_gaze
        
        result = self.gaze_tracker.get_gaze_direction()
        assert result["direction"] == "down"
    
    def test_get_gaze_direction_combined(self):
        """Тест получения комбинированного направления взгляда"""
        self.gaze_tracker.calibrated = True
        self.gaze_tracker.horizontal_center = 0.5
        self.gaze_tracker.vertical_center = 0.5
        self.gaze_tracker.calibration_threshold = 0.1
        
        mock_gaze = Mock()
        mock_gaze.horizontal_ratio.return_value = 0.7  
        mock_gaze.vertical_ratio.return_value = 0.3   
        self.gaze_tracker.gaze = mock_gaze
        
        result = self.gaze_tracker.get_gaze_direction()
        assert "left" in result["direction"]
        assert "up" in result["direction"]
    
    def test_get_eye_position(self):
        """Тест получения координат глаз"""
        mock_gaze = Mock()
        mock_gaze.pupil_left_coords.return_value = (100, 200)
        mock_gaze.pupil_right_coords.return_value = (300, 200)
        self.gaze_tracker.gaze = mock_gaze
        
        left_coords, right_coords = self.gaze_tracker.get_eye_position()
        
        assert left_coords == (100, 200)
        assert right_coords == (300, 200)
    
    @patch('cv2.destroyAllWindows')
    def test_release_camera(self, mock_destroy):
        """Тест освобождения камеры"""
        mock_camera = Mock()
        self.gaze_tracker.camera = mock_camera
        self.gaze_tracker.debug = True
        
        self.gaze_tracker.release_camera()
        
        mock_camera.release.assert_called_once()
        mock_destroy.assert_called_once()
    
    @patch('cv2.destroyAllWindows')
    def test_release_camera_no_camera(self, mock_destroy):
        """Тест освобождения камеры без инициализации"""
        self.gaze_tracker.camera = None
        self.gaze_tracker.debug = True
        
        self.gaze_tracker.release_camera()
        
        mock_destroy.assert_called_once()
    
    def test_release_camera_no_debug(self):
        """Тест освобождения камеры без отладочного режима"""
        mock_camera = Mock()
        self.gaze_tracker.camera = mock_camera
        self.gaze_tracker.debug = False
        
        with patch('cv2.destroyAllWindows') as mock_destroy:
            self.gaze_tracker.release_camera()
            mock_destroy.assert_not_called() 