import cv2
from gaze_tracking import GazeTracking
import time
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path


def load_config(config_path: str = "config.json") -> Dict:
    default_config = {
        "debug": True,
        "debug_window_size": [800, 600],
        "calibration_threshold": 0.10,
        "max_suspicious_actions": 10,
        "logs_dir": "logs",
        "gaze_log_file": "gaze_log.txt",
        "behavior_log_file": "behavior_log.json",
        "calibration_time": 10,
        "analysis_window": 20,
        "sleep_interval": 0.1
    }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # Объединяем с дефолтными значениями
        return {**default_config, **config}
    except FileNotFoundError:
        print(f"Файл конфигурации {config_path} не найден, используются значения по умолчанию")
        return default_config


CONFIG = load_config()

Path(CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)


class GazeTracker:
    def __init__(self, debug: bool = True, calibration_threshold: float = 0.10):
        self.gaze = GazeTracking()
        self.camera = None
        self.debug = CONFIG["debug"] if debug is None else debug
        self.debug_window_size = tuple(CONFIG["debug_window_size"])
        self.calibration_threshold = CONFIG[
            "calibration_threshold"] if calibration_threshold is None else calibration_threshold
        self.horizontal_center = 0.5
        self.vertical_center = 0.5
        self.calibrated = False
        self.calibration_time = CONFIG["calibration_time"]

    def initialize_camera(self) -> None:
        """Инициализация камеры с калибровкой."""
        self.camera = cv2.VideoCapture(0)
        self.calibrate()

    def detect_gaze(self) -> Optional[str]:
        """Определение направления взгляда с отладочным выводом"""
        if self.camera is None:
            raise ValueError("Камера не инициализирована.")

        _, frame = self.camera.read()
        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()

        gaze_info = self.get_gaze_direction()
        direction = gaze_info["direction"]

        # Отладочный вывод
        if self.debug:
            debug_frame = frame.copy()
            debug_frame = cv2.resize(debug_frame, self.debug_window_size)

            cv2.putText(debug_frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"H: {gaze_info.get('horizontal_ratio', 0):.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"V: {gaze_info.get('vertical_ratio', 0):.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Debug: Eye Tracking", debug_frame)
            cv2.waitKey(1)

        return direction if direction != "not calibrated" else None

    def get_eye_position(self) -> Tuple[int, int]:
        """Получение координат глаз."""
        return self.gaze.pupil_left_coords(), self.gaze.pupil_right_coords()

    def release_camera(self) -> None:
        """Освобождение камеры и закрытие окон"""
        if self.camera is not None:
            self.camera.release()
        if self.debug:
            cv2.destroyAllWindows()

    def calibrate(self) -> None:
        """Калибровка центрального положения глаз."""
        print("Калибровка: направьте глаза в центр экрана и нажмите 'c'")
        while True:
            _, frame = self.camera.read()
            self.gaze.refresh(frame)
            frame = self.gaze.annotated_frame()

            debug_frame = frame.copy()
            debug_frame = cv2.resize(debug_frame, self.debug_window_size)
            cv2.putText(debug_frame, "Calibration: Look straight and press 'c'",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", debug_frame)

            key = cv2.waitKey(1)
            if key == ord('c') or key == ord('с'):
                break

        print("Калибровка... Не двигайте глазами 10 секунд")
        start_time = time.time()
        horizontal_values = []
        vertical_values = []

        while time.time() - start_time < self.calibration_time:
            _, frame = self.camera.read()
            self.gaze.refresh(frame)

            if self.gaze.horizontal_ratio() is not None:
                horizontal_values.append(self.gaze.horizontal_ratio())
            if self.gaze.vertical_ratio() is not None:
                vertical_values.append(self.gaze.vertical_ratio())

            time.sleep(0.1)

        if horizontal_values and vertical_values:
            self.horizontal_center = sum(horizontal_values) / len(horizontal_values)
            self.vertical_center = sum(vertical_values) / len(vertical_values)
            self.calibrated = True
            print(f"Калибровка завершена. Центр: H={self.horizontal_center:.2f}, V={self.vertical_center:.2f}")
        else:
            print("Ошибка калибровки. Используются значения по умолчанию.")
        cv2.destroyAllWindows()

    def get_gaze_direction(self) -> Dict[str, any]:
        """Определение направления взгляда относительно калиброванного центра."""
        if not self.calibrated:
            return {"direction": "not calibrated"}

        horizontal = self.gaze.horizontal_ratio()
        vertical = self.gaze.vertical_ratio()

        if horizontal is None or vertical is None:
            return {"direction": "blink"}

        h_diff = horizontal - self.horizontal_center
        v_diff = vertical - self.vertical_center

        direction = []

        if abs(h_diff) > self.calibration_threshold:
            direction.append("right" if h_diff < 0 else "left")
        if abs(v_diff) > self.calibration_threshold:
            direction.append("up" if v_diff < 0 else "down")

        if not direction:
            direction.append("center")

        return {
            "direction": " ".join(direction),
            "horizontal_ratio": horizontal,
            "vertical_ratio": vertical
        }


class BehaviorAnalyzer:
    def __init__(self, max_suspicious_actions: int = 10):
        self.suspicious_actions = 0
        self.max_suspicious_actions = CONFIG[
            "max_suspicious_actions"] if max_suspicious_actions is None else max_suspicious_actions
        self.gaze_history = []
        self.analysis_window = CONFIG["analysis_window"]

    def analyze_gaze_pattern(self, gaze_data: str) -> None:
        """Анализ паттернов взгляда на основе калибровки."""
        self.gaze_history.append(gaze_data)

        if gaze_data not in ["center", "blink", "not calibrated"]:
            self.suspicious_actions += 1
        elif self.suspicious_actions > 0:
            self.suspicious_actions -= 1

    def detect_cheating(self) -> bool:
        """Проверка на списывание."""
        return self.suspicious_actions >= self.max_suspicious_actions

    def generate_report(self) -> Dict[str, any]:
        """Генерация отчета."""
        report = {
            "suspicious_actions": self.suspicious_actions,
            "gaze_history": self.gaze_history[-self.analysis_window:],
            "current_status": "cheating" if self.detect_cheating() else "normal"
        }
        self.suspicious_actions = 0
        return report


class UIInterface:
    @staticmethod
    def display_gaze_data(gaze_data: str) -> None:
        """Отображение направления взгляда."""
        #(f"Направление взгляда: {gaze_data}")

    @staticmethod
    def show_alert() -> None:
        """Предупреждение о списывании."""
        #print("Внимание! Обнаружено подозрительное поведение!")

    @staticmethod
    def display_report(report: Dict[str, any]) -> None:
        """Отображение отчета."""
        #print("Отчет о поведении:")
        #print(json.dumps(report, indent=2))


class DataLogger:
    def __init__(self):
        self.gaze_logs = []
        self.behavior_logs = []
        self.logs_dir = CONFIG["logs_dir"]
        self.gaze_log_file = Path(self.logs_dir) / CONFIG["gaze_log_file"]
        self.behavior_log_file = Path(self.logs_dir) / CONFIG["behavior_log_file"]

    def log_gaze_data(self, gaze_data: str) -> None:
        """Логирование данных о взгляде в память."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.gaze_logs.append(f"{timestamp}: {gaze_data}")

    def log_behavior(self, behavior_data: Dict[str, any]) -> None:
        """Логирование данных о поведении в память."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = {
            "timestamp": timestamp,
            "data": behavior_data
        }
        self.behavior_logs.append(log_entry)

    def save_logs_to_file(self) -> None:
        """Сохранение логов из памяти в файлы."""
        if self.gaze_logs:
            with open(self.gaze_log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(self.gaze_logs) + "\n")
            self.gaze_logs = []


        if self.behavior_logs:
            try:
                if self.behavior_log_file.exists():
                    with open(self.behavior_log_file, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                else:
                    existing_logs = []
            except json.JSONDecodeError:
                existing_logs = []

            existing_logs.extend(self.behavior_logs)

            with open(self.behavior_log_file, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            self.behavior_logs = []



class MainApp:
    def __init__(self):
        self.gaze_tracker = GazeTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.ui = UIInterface()
        self.logger = DataLogger()
        self.sleep_interval = CONFIG["sleep_interval"]

    def run(self) -> None:
        """Запуск приложения."""
        try:
            participant_number = input("Введите номер участника: ")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            gaze_log_entry = f"{timestamp}: Номер участника: {participant_number}"
            self.logger.gaze_logs.append(gaze_log_entry)

            behavior_log_entry = {
                "timestamp": timestamp,
                "data": {
                    "participant_number": participant_number,
                    "message": "Начало сеанса, номер участника записан"
                }
            }
            self.logger.behavior_logs.append(behavior_log_entry)
            self.logger.save_logs_to_file()

            self.gaze_tracker.initialize_camera()
            print("Калибровка завершена. Приложение запущено. Нажмите C для остановки.")

            while True:
                gaze_data = self.gaze_tracker.detect_gaze()
                if gaze_data and gaze_data != "not calibrated":
                    self.ui.display_gaze_data(gaze_data)
                    self.behavior_analyzer.analyze_gaze_pattern(gaze_data)
                    self.logger.log_gaze_data(gaze_data)

                    if self.behavior_analyzer.detect_cheating():
                        self.ui.show_alert()
                        report = self.behavior_analyzer.generate_report()
                        self.logger.log_behavior(report)
                        self.ui.display_report(report)

                time.sleep(self.sleep_interval)

                key = cv2.waitKey(1)
                if key == ord('x') or key == ord('ч'):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    self.logger.gaze_logs.append(f"{timestamp}: Отмечена попытка списывания (по нажатию X)")
                    self.logger.behavior_logs.append({
                        "timestamp": timestamp,
                        "data": {
                            "event_type": "manual_cheating_mark",
                            "message": "Пользователь отметил попытку списывания по нажатию X"
                        }
                    })
                    self.logger.save_logs_to_file()
                    print("Попытка списывания отмечена в логах")

                if key == ord('c') or key == ord('с'):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Остановка приложения."""
        self.gaze_tracker.release_camera()
        self.logger.save_logs_to_file()
        print("Общий лог активности успешно сохранен!")
        print("Лог подозрительной активности успешно сохранен!")
        print("Приложение остановлено.")


if __name__ == "__main__":
    app = MainApp()
    app.run()