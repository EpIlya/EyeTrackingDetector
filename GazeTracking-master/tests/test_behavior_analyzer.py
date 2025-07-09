import pytest
import time
from unittest.mock import Mock, patch
from main import BehaviorAnalyzer, CONFIG


class TestBehaviorAnalyzer:
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.analyzer = BehaviorAnalyzer()
    
    def test_initialization(self):
        """Тест инициализации BehaviorAnalyzer"""
        assert self.analyzer.suspicious_actions == 0
        assert self.analyzer.gaze_history == []
        assert self.analyzer.consecutive_offcenter == 0
        assert self.analyzer.last_direction == "center"
    
    def test_analyze_gaze_pattern_center(self):
        """Тест анализа взгляда в центр"""
        self.analyzer.analyze_gaze_pattern("center")
        assert self.analyzer.consecutive_offcenter == -1
        assert len(self.analyzer.gaze_history) == 1
        assert self.analyzer.gaze_history[0][1] == "center"
    
    def test_analyze_gaze_pattern_offcenter(self):
        """Тест анализа взгляда вне центра"""
        self.analyzer.analyze_gaze_pattern("left")
        assert self.analyzer.consecutive_offcenter == 1
        assert len(self.analyzer.gaze_history) == 1
        assert self.analyzer.gaze_history[0][1] == "left"
    
    def test_consecutive_offcenter_threshold(self):
        """Тест превышения порога подряд идущих кадров вне центра"""
        for _ in range(self.analyzer.min_consecutive_offcenter):
            self.analyzer.analyze_gaze_pattern("left")

        assert self.analyzer.suspicious_actions >= 1
        assert self.analyzer.consecutive_offcenter == 0
    
    def test_blink_reset_consecutive(self):
        """Тест сброса счетчика при моргании"""
        self.analyzer.analyze_gaze_pattern("left")
        assert self.analyzer.consecutive_offcenter == 1

        self.analyzer.analyze_gaze_pattern("blink")
        assert self.analyzer.consecutive_offcenter == 0
    
    def test_detect_cheating_false(self):
        """Тест отсутствия списывания при нормальном поведении"""
        assert not self.analyzer.detect_cheating()
    
    def test_detect_cheating_true(self):
        """Тест обнаружения списывания"""
        self.analyzer.suspicious_actions = self.analyzer.max_suspicious_actions
        assert self.analyzer.detect_cheating()
    
    def test_generate_report(self):
        """Тест генерации отчета"""
        self.analyzer.analyze_gaze_pattern("left")
        self.analyzer.analyze_gaze_pattern("right")
        
        report = self.analyzer.generate_report()
        
        assert "suspicious_actions" in report
        assert "gaze_history" in report
        assert "current_status" in report
        assert len(report["gaze_history"]) == 2
        assert report["current_status"] in ["normal", "cheating"]
    
    def test_window_size_calculation(self):
        """Тест расчета размера окна"""
        expected_window_size = int(CONFIG["analysis_window"] / CONFIG["sleep_interval"])
        assert self.analyzer.window_size == expected_window_size
    
    def test_min_consecutive_offcenter_calculation(self):
        """Тест расчета минимального количества кадров вне центра"""
        expected_min_consecutive = int(2.0 / CONFIG["sleep_interval"])
        assert self.analyzer.min_consecutive_offcenter == expected_min_consecutive
    
    def test_offcenter_threshold_analysis(self):
        """Тест анализа процента времени вне центра"""
        window_size = self.analyzer.window_size
        offcenter_threshold = self.analyzer.offcenter_threshold

        offcenter_count = int(window_size * offcenter_threshold) + 1

        for _ in range(offcenter_count):
            self.analyzer.analyze_gaze_pattern("left")

        assert self.analyzer.suspicious_actions >= 1

        remaining_count = window_size - offcenter_count
        for _ in range(remaining_count):
            self.analyzer.analyze_gaze_pattern("center")

        expected_size = window_size // 2
        assert len(self.analyzer.gaze_history) == expected_size
    
    def test_center_gaze_decrease_suspicious_actions(self):
        """Тест уменьшения счетчика при возврате в центр"""
        self.analyzer.suspicious_actions = 5

        self.analyzer.analyze_gaze_pattern("center")

        assert self.analyzer.suspicious_actions == 4
    
    def test_history_overflow(self):
        """Тест переполнения истории взгляда"""
        for i in range(self.analyzer.window_size + 10):
            self.analyzer.analyze_gaze_pattern("left")

        assert len(self.analyzer.gaze_history) <= self.analyzer.window_size
    
    def test_negative_consecutive_offcenter(self):
        """Тест отрицательного значения consecutive_offcenter"""
        for _ in range(3):
            self.analyzer.analyze_gaze_pattern("center")

        assert self.analyzer.consecutive_offcenter == -3

        self.analyzer.analyze_gaze_pattern("left")
        assert self.analyzer.consecutive_offcenter == -2 