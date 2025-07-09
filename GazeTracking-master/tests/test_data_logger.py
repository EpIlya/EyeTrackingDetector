import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from main import DataLogger, CONFIG


class TestDataLogger:
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.logger = DataLogger()
    
    def test_initialization(self):
        """Тест инициализации DataLogger"""
        assert self.logger.gaze_logs == []
        assert self.logger.behavior_logs == []
        assert self.logger.logs_dir == CONFIG["logs_dir"]
        assert self.logger.gaze_log_file == Path(CONFIG["logs_dir"]) / CONFIG["gaze_log_file"]
        assert self.logger.behavior_log_file == Path(CONFIG["logs_dir"]) / CONFIG["behavior_log_file"]
    
    def test_log_gaze_data(self):
        """Тест логирования данных о взгляде"""
        gaze_data = "center"
        initial_length = len(self.logger.gaze_logs)
        
        self.logger.log_gaze_data(gaze_data)
        
        assert len(self.logger.gaze_logs) == initial_length + 1
        assert self.logger.gaze_logs[-1].endswith(f": {gaze_data}")
    
    def test_log_behavior(self):
        """Тест логирования данных о поведении"""
        behavior_data = {"suspicious_actions": 5, "status": "normal"}
        initial_length = len(self.logger.behavior_logs)
        
        self.logger.log_behavior(behavior_data)
        
        assert len(self.logger.behavior_logs) == initial_length + 1
        assert "timestamp" in self.logger.behavior_logs[-1]
        assert self.logger.behavior_logs[-1]["data"] == behavior_data
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=False)
    def test_save_logs_to_file_empty_logs(self, mock_exists, mock_file):
        """Тест сохранения пустых логов"""
        self.logger.save_logs_to_file()

        mock_file.assert_not_called()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=False)
    def test_save_gaze_logs_to_file(self, mock_exists, mock_file):
        """Тест сохранения логов взгляда"""
        self.logger.gaze_logs = ["2023-01-01 12:00:00: center", "2023-01-01 12:00:01: left"]
        
        self.logger.save_logs_to_file()

        mock_file.assert_called_once()
        assert "a" in mock_file.call_args[0]

        assert self.logger.gaze_logs == []
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('json.load', return_value=[{"existing": "data"}])
    def test_save_behavior_logs_to_file_existing(self, mock_json_load, mock_exists, mock_file):
        """Тест сохранения логов поведения с существующим файлом"""
        behavior_data = {"suspicious_actions": 5}
        self.logger.behavior_logs = [{"timestamp": "2023-01-01 12:00:00", "data": behavior_data}]
        
        self.logger.save_logs_to_file()

        assert mock_file.call_count >= 2

        assert self.logger.behavior_logs == []
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=False)
    def test_save_behavior_logs_to_file_new(self, mock_exists, mock_file):
        """Тест сохранения логов поведения в новый файл"""
        behavior_data = {"suspicious_actions": 5}
        self.logger.behavior_logs = [{"timestamp": "2023-01-01 12:00:00", "data": behavior_data}]
        
        self.logger.save_logs_to_file()

        mock_file.assert_called()

        assert self.logger.behavior_logs == []
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_save_behavior_logs_to_file_invalid_json(self, mock_json_load, mock_exists, mock_file):
        """Тест сохранения логов поведения с некорректным JSON"""
        behavior_data = {"suspicious_actions": 5}
        self.logger.behavior_logs = [{"timestamp": "2023-01-01 12:00:00", "data": behavior_data}]
        
        self.logger.save_logs_to_file()

        mock_file.assert_called()

        assert self.logger.behavior_logs == []
    
    def test_log_multiple_gaze_data(self):
        """Тест логирования множественных данных о взгляде"""
        gaze_data_list = ["center", "left", "right", "up", "down"]
        
        for data in gaze_data_list:
            self.logger.log_gaze_data(data)
        
        assert len(self.logger.gaze_logs) == len(gaze_data_list)
        for i, data in enumerate(gaze_data_list):
            assert self.logger.gaze_logs[i].endswith(f": {data}")
    
    def test_log_multiple_behavior_data(self):
        """Тест логирования множественных данных о поведении"""
        behavior_data_list = [
            {"suspicious_actions": 1, "status": "normal"},
            {"suspicious_actions": 5, "status": "warning"},
            {"suspicious_actions": 10, "status": "cheating"}
        ]
        
        for data in behavior_data_list:
            self.logger.log_behavior(data)
        
        assert len(self.logger.behavior_logs) == len(behavior_data_list)
        for i, data in enumerate(behavior_data_list):
            assert self.logger.behavior_logs[i]["data"] == data
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=False)
    def test_save_logs_to_file_both_types(self, mock_exists, mock_file):
        """Тест сохранения обоих типов логов одновременно"""
        self.logger.gaze_logs = ["2023-01-01 12:00:00: center"]
        self.logger.behavior_logs = [{"timestamp": "2023-01-01 12:00:00", "data": {"test": "data"}}]
        
        self.logger.save_logs_to_file()

        assert mock_file.call_count >= 2

        assert self.logger.gaze_logs == []
        assert self.logger.behavior_logs == []
    
    def test_timestamp_format(self):
        """Тест формата временной метки"""
        self.logger.log_gaze_data("center")
        
        timestamp = self.logger.gaze_logs[0].split(": ")[0]
        assert len(timestamp) == 19
        assert timestamp.count("-") == 2
        assert timestamp.count(":") == 2
        assert timestamp.count(" ") == 1
    
    def test_behavior_log_structure(self):
        """Тест структуры лога поведения"""
        behavior_data = {"suspicious_actions": 5, "status": "normal"}
        self.logger.log_behavior(behavior_data)
        
        log_entry = self.logger.behavior_logs[0]
        assert "timestamp" in log_entry
        assert "data" in log_entry
        assert log_entry["data"] == behavior_data 