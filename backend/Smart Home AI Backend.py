from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import datetime
import logging
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Flask и CORS
app = Flask(__name__)
CORS(app)  # Разрешаем кросс-доменные запросы

# Инициализация стеммера для русского языка
stemmer = SnowballStemmer("russian")

# Классы команд
COMMAND_CLASSES = {
    'roof_open': ['старт', 'открой крышу', 'открыть крышу', 'шатырды аш'],
    'roof_close': ['стоп', 'закрой крышу', 'закрыть крышу', 'шатырды жап'],
    'garage_open': ['открой гараж', 'открыть гараж', 'гаражды аш'],
    'garage_close': ['закрой гараж', 'закрыть гараж', 'гаражды жап'],
    'sprinkler_on': ['включи полив', 'включить полив', 'суаруды қос'],
    'sprinkler_off': ['выключи полив', 'выключить полив', 'суаруды өшір'],
    'light_on_living': ['включи свет в зале', 'зажги свет в гостиной', 'қонақ бөлмедегі жарықты қос'],
    'light_off_living': ['выключи свет в зале', 'погаси свет в гостиной', 'қонақ бөлмедегі жарықты өшір'],
    'tv_on_living': ['включи телевизор в зале', 'теледидарды қонақ бөлмеде қос'],
    'tv_off_living': ['выключи телевизор в зале', 'теледидарды қонақ бөлмеде өшір'],
    'coffee_on_kitchen': ['включи кофе в кухне', 'ас бөлмеде кофені қос'],
    'coffee_off_kitchen': ['выключи кофе в кухне', 'ас бөлмеде кофені өшір'],
    'bed_temp_up': ['сделай кровать теплее', 'жатын бөлмеде төсек жылырақ жаса'],
    'bed_temp_down': ['сделай кровать прохладнее', 'жатын бөлмеде төсек салқын жаса'],
    'projector_on_office': ['включи проектор в кабинете', 'жұмыс бөлмеде проекторды қос'],
    'projector_off_office': ['выключи проектор в кабинете', 'жұмыс бөлмеде проекторды өшір'],
    'vr_on_gaming': ['включи VR в игровой', 'ойын бөлмеде VR-ды қос'],
    'vr_off_gaming': ['выключи VR в игровой', 'ойын бөлмеде VR-ды өшір'],
    # Добавьте другие команды по аналогии
}

# Словарь с комнатами
ROOMS = {
    'living_room': ['зал', 'гостиная', 'қонақ бөлме'],
    'kitchen': ['кухня', 'ас бөлме'],
    'bedroom': ['спальня', 'жатын бөлме'],
    'office': ['кабинет', 'жұмыс бөлме'],
    'gaming_room': ['игровая', 'ойын бөлме'],
}

# Путь к моделям
MODEL_PATH = 'models/'
COMMAND_MODEL_PATH = os.path.join(MODEL_PATH, 'command_classifier.pkl')
VECTORIZER_PATH = os.path.join(MODEL_PATH, 'vectorizer.pkl')
USER_PROFILES_PATH = os.path.join(MODEL_PATH, 'user_profiles.json')
COMMAND_HISTORY_PATH = os.path.join(MODEL_PATH, 'command_history.json')

# Создание директории для моделей, если она не существует
os.makedirs(MODEL_PATH, exist_ok=True)

# Класс для модели ИИ
class SmartHomeAI:
    def __init__(self):
        self.vectorizer = None
        self.command_classifier = None
        self.user_profiles = {}
        self.command_history = {}
        
        # Загрузка моделей и данных при инициализации
        self.load_models()
        
    def load_models(self):
        """Загрузка моделей из файлов или создание новых моделей"""
        try:
            # Пытаемся загрузить модель классификатора команд
            if os.path.exists(COMMAND_MODEL_PATH):
                with open(COMMAND_MODEL_PATH, 'rb') as f:
                    self.command_classifier = pickle.load(f)
                logger.info("Модель классификатора команд загружена успешно")
            else:
                # Если модель не существует, создаем новый классификатор
                self.command_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                logger.info("Создан новый классификатор команд")
            
            # Загрузка векторизатора
            if os.path.exists(VECTORIZER_PATH):
                with open(VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Векторизатор загружен успешно")
            else:
                # Если векторизатор не существует, создаем новый
                self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize_and_stem)
                logger.info("Создан новый векторизатор")
            
            # Загрузка профилей пользователей
            if os.path.exists(USER_PROFILES_PATH):
                with open(USER_PROFILES_PATH, 'r', encoding='utf-8') as f:
                    self.user_profiles = json.load(f)
                logger.info("Профили пользователей загружены успешно")
            
            # Загрузка истории команд
            if os.path.exists(COMMAND_HISTORY_PATH):
                with open(COMMAND_HISTORY_PATH, 'r', encoding='utf-8') as f:
                    self.command_history = json.load(f)
                logger.info("История команд загружена успешно")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {str(e)}")
            # Если произошла ошибка при загрузке, создаем новые модели
            self.command_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize_and_stem)
            logger.info("Созданы новые модели из-за ошибки загрузки")
    
    def save_models(self):
        """Сохранение моделей в файлы"""
        try:
            # Сохраняем классификатор команд
            with open(COMMAND_MODEL_PATH, 'wb') as f:
                pickle.dump(self.command_classifier, f)
            
            # Сохраняем векторизатор
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Сохраняем профили пользователей
            with open(USER_PROFILES_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles, f, ensure_ascii=False, indent=4)
            
            # Сохраняем историю команд
            with open(COMMAND_HISTORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.command_history, f, ensure_ascii=False, indent=4)
                
            logger.info("Модели и данные успешно сохранены")
        except Exception as e:
            logger.error(f"Ошибка сохранения моделей: {str(e)}")
    
    def tokenize_and_stem(self, text):
        """Токенизация и стемминг текста"""
        tokens = word_tokenize(text.lower(), language='russian')
        # Удаление стоп-слов и пунктуации
        filtered_tokens = [token for token in tokens if token.isalpha()]
        # Стемминг
        stems = [stemmer.stem(token) for token in filtered_tokens]
        return stems
    
    def train_command_classifier(self):
        """Обучение классификатора команд"""
        try:
            # Подготовка данных для обучения
            X = []
            y = []
            
            for command_class, examples in COMMAND_CLASSES.items():
                for example in examples:
                    X.append(example)
                    y.append(command_class)
            
            # Обучение векторизатора
            X_vectors = self.vectorizer.fit_transform(X)
            
            # Обучение классификатора
            self.command_classifier.fit(X_vectors, y)
            
            logger.info("Классификатор команд успешно обучен")
            
            # Сохранение моделей после обучения
            self.save_models()
        except Exception as e:
            logger.error(f"Ошибка обучения классификатора: {str(e)}")
    
    def classify_command(self, command_text):
        """Классификация команды на основе текста"""
        try:
            # Если модель еще не обучена, обучаем ее
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.train_command_classifier()
            
            # Векторизация команды
            command_vector = self.vectorizer.transform([command_text])
            
            # Вероятности классов
            class_probabilities = self.command_classifier.predict_proba(command_vector)[0]
            classes = self.command_classifier.classes_
            
            # Получаем наиболее вероятный класс и его вероятность
            max_prob_index = np.argmax(class_probabilities)
            predicted_class = classes[max_prob_index]
            confidence = class_probabilities[max_prob_index]
            
            # Определение комнаты, если указана
            room = self.extract_room(command_text)
            
            logger.info(f"Команда '{command_text}' классифицирована как '{predicted_class}' с уверенностью {confidence:.2f}")
            
            return {
                'command_class': predicted_class,
                'confidence': float(confidence),
                'room': room
            }
        except Exception as e:
            logger.error(f"Ошибка классификации команды: {str(e)}")
            return {
                'command_class': 'unknown',
                'confidence': 0.0,
                'room': None
            }
    
    def extract_room(self, command_text):
        """Извлечение комнаты из текста команды"""
        command_lower = command_text.lower()
        
        for room_key, room_aliases in ROOMS.items():
            for alias in room_aliases:
                if alias in command_lower:
                    return room_key
        
        return None
    
    def log_command(self, user_id, command_text, predicted_class, room=None):
        """Запись команды в историю"""
        timestamp = datetime.datetime.now().isoformat()
        
        if user_id not in self.command_history:
            self.command_history[user_id] = []
        
        self.command_history[user_id].append({
            'command': command_text,
            'predicted_class': predicted_class,
            'room': room,
            'timestamp': timestamp
        })
        
        # Обновляем профиль пользователя
        self.update_user_profile(user_id, predicted_class, room)
        
        # Сохраняем обновленные данные
        self.save_models()
    
    def update_user_profile(self, user_id, command_class, room=None):
        """Обновление профиля пользователя на основе его команд"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'command_frequencies': {},
                'room_frequencies': {},
                'time_patterns': {
                    'morning': 0,   # 5-12
                    'afternoon': 0, # 12-17
                    'evening': 0,   # 17-22
                    'night': 0      # 22-5
                }
            }
        
        profile = self.user_profiles[user_id]
        
        # Обновляем частоту команд
        if command_class not in profile['command_frequencies']:
            profile['command_frequencies'][command_class] = 0
        profile['command_frequencies'][command_class] += 1
        
        # Обновляем частоту комнат
        if room:
            if room not in profile['room_frequencies']:
                profile['room_frequencies'][room] = 0
            profile['room_frequencies'][room] += 1
        
        # Обновляем временные паттерны
        current_hour = datetime.datetime.now().hour
        if 5 <= current_hour < 12:
            profile['time_patterns']['morning'] += 1
        elif 12 <= current_hour < 17:
            profile['time_patterns']['afternoon'] += 1
        elif 17 <= current_hour < 22:
            profile['time_patterns']['evening'] += 1
        else:
            profile['time_patterns']['night'] += 1
    
    def predict_next_action(self, user_id):
        """Предсказание следующего действия пользователя на основе его профиля и истории"""
        if user_id not in self.user_profiles or user_id not in self.command_history:
            return None
        
        profile = self.user_profiles[user_id]
        history = self.command_history[user_id]
        
        if not history or not profile['command_frequencies']:
            return None
        
        # Получаем текущий час
        current_hour = datetime.datetime.now().hour
        
        # Определяем время суток
        time_of_day = None
        if 5 <= current_hour < 12:
            time_of_day = 'morning'
        elif 12 <= current_hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= current_hour < 22:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        
        # Находим наиболее частые команды для текущего времени суток
        command_counts = {}
        time_filtered_history = [entry for entry in history 
                                if datetime.datetime.fromisoformat(entry['timestamp']).hour in 
                                self._get_hour_range(time_of_day)]
        
        if not time_filtered_history:
            # Если нет истории для текущего времени, используем общую частоту команд
            for command, count in profile['command_frequencies'].items():
                command_counts[command] = count
        else:
            # Подсчитываем частоту команд для текущего времени суток
            for entry in time_filtered_history:
                command = entry['predicted_class']
                if command not in command_counts:
                    command_counts[command] = 0
                command_counts[command] += 1
        
        # Находим наиболее частые комнаты
        room_weights = profile['room_frequencies'].copy() if profile['room_frequencies'] else {'bedroom': 1}
        
        # Нормализуем веса комнат
        total_room_weight = sum(room_weights.values())
        if total_room_weight > 0:
            for room in room_weights:
                room_weights[room] /= total_room_weight
        
        # Сортируем команды по частоте
        sorted_commands = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_commands:
            return None
        
        # Выбираем самую частую команду
        predicted_command = sorted_commands[0][0]
        
        # Выбираем комнату с наибольшим весом
        predicted_room = max(room_weights.items(), key=lambda x: x[1])[0] if room_weights else None
        
        return {
            'command_class': predicted_command,
            'room': predicted_room,
            'time_of_day': time_of_day
        }

    def _get_hour_range(self, time_of_day):
        """Получение диапазона часов для заданного времени суток"""
        if time_of_day == 'morning':
            return range(5, 12)
        elif time_of_day == 'afternoon':
            return range(12, 17)
        elif time_of_day == 'evening':
            return range(17, 22)
        else:  # night
            return list(range(22, 24)) + list(range(0, 5))
    
    def get_suggestions(self, user_id):
        """Получение рекомендаций для пользователя"""
        prediction = self.predict_next_action(user_id)
        if not prediction:
            return []
        
        suggestions = []
        
        command_class = prediction['command_class']
        room = prediction['room']
        
        # Базовые рекомендации на основе предсказанного действия
        if command_class.startswith('light_on'):
            suggestions.append(f"Включить свет в {self._get_room_name(room)}")
        elif command_class.startswith('light_off'):
            suggestions.append(f"Выключить свет в {self._get_room_name(room)}")
        elif command_class.startswith('door_open'):
            suggestions.append(f"Открыть дверь в {self._get_room_name(room)}")
        elif command_class.startswith('door_close'):
            suggestions.append(f"Закрыть дверь в {self._get_room_name(room)}")
        elif command_class.startswith('window_open'):
            suggestions.append(f"Открыть окно в {self._get_room_name(room)}")
        elif command_class.startswith('window_close'):
            suggestions.append(f"Закрыть окно в {self._get_room_name(room)}")
        elif command_class == 'garage_open':
            suggestions.append("Открыть гараж")
        elif command_class == 'garage_close':
            suggestions.append("Закрыть гараж")
        elif command_class == 'sprinkler_on':
            suggestions.append("Включить полив")
        elif command_class == 'sprinkler_off':
            suggestions.append("Выключить полив")
        
        # Добавляем случайные рекомендации из общих команд
        general_commands = [
            "Включить весь свет",
            "Выключить весь свет",
            "Открыть крышу",
            "Закрыть крышу"
        ]
        
        random_suggestions = random.sample(general_commands, min(2, len(general_commands)))
        suggestions.extend(random_suggestions)
        
        return suggestions[:3]  # Возвращаем не более 3 рекомендаций
    
    def _get_room_name(self, room_key):
        """Получение названия комнаты на русском языке"""
        room_names = {
            'bedroom': 'спальне',
            'kitchen': 'кухне',
            'living_room': 'зале',
            'hall': 'холле',
            'entrance': 'прихожей',
            'corridor': 'коридоре'
        }
        return room_names.get(room_key, 'комнате')

# Инициализация AI
smart_home_ai = SmartHomeAI()

# Маршруты API
@app.route('/api/process-command', methods=['POST'])
def process_command():
    """Обработка команды от пользователя"""
    try:
        data = request.json
        command_text = data.get('command', '')
        user_id = data.get('user_id', 'anonymous')
        
        if not command_text:
            return jsonify({'error': 'Команда не указана'}), 400
        
        # Классификация команды
        result = smart_home_ai.classify_command(command_text)
        
        # Записываем команду в историю
        smart_home_ai.log_command(
            user_id, 
            command_text, 
            result['command_class'], 
            result['room']
        )
        
        return jsonify({
            'command_class': result['command_class'],
            'confidence': result['confidence'],
            'room': result['room'],
            'processed': True
        })
    except Exception as e:
        logger.error(f"Ошибка обработки команды: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Получение предложений для пользователя"""
    try:
        user_id = request.args.get('user_id', 'anonymous')
        
        # Получаем предложения для пользователя
        suggestions = smart_home_ai.get_suggestions(user_id)
        
        return jsonify({
            'suggestions': suggestions
        })
    except Exception as e:
        logger.error(f"Ошибка получения предложений: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Получение аналитики по командам"""
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            # Общая аналитика по всем пользователям
            all_commands = []
            for user, commands in smart_home_ai.command_history.items():
                all_commands.extend(commands)
                
            # Подсчитываем статистику команд
            command_stats = {}
            for entry in all_commands:
                command = entry['predicted_class']
                if command not in command_stats:
                    command_stats[command] = 0
                command_stats[command] += 1
            
            # Подсчитываем статистику по комнатам
            room_stats = {}
            for entry in all_commands:
                room = entry['room']
                if room:
                    if room not in room_stats:
                        room_stats[room] = 0
                    room_stats[room] += 1
            
            return jsonify({
                'command_stats': command_stats,
                'room_stats': room_stats,
                'total_commands': len(all_commands)
            })
        else:
            # Аналитика для конкретного пользователя
            if user_id not in smart_home_ai.command_history:
                return jsonify({'error': 'Пользователь не найден'}), 404
                
            user_commands = smart_home_ai.command_history[user_id]
            
            # Подсчитываем статистику команд пользователя
            command_stats = {}
            for entry in user_commands:
                command = entry['predicted_class']
                if command not in command_stats:
                    command_stats[command] = 0
                command_stats[command] += 1
            
            # Подсчитываем статистику по комнатам для пользователя
            room_stats = {}
            for entry in user_commands:
                room = entry['room']
                if room:
                    if room not in room_stats:
                        room_stats[room] = 0
                    room_stats[room] += 1
            
            # Получаем профиль пользователя
            user_profile = smart_home_ai.user_profiles.get(user_id, {})
            
            return jsonify({
                'command_stats': command_stats,
                'room_stats': room_stats,
                'time_patterns': user_profile.get('time_patterns', {}),
                'total_commands': len(user_commands)
            })
    except Exception as e:
        logger.error(f"Ошибка получения аналитики: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Обучение модели при запуске
@app.before_request
def before_request_func():
    print("Этот код выполнится перед каждым запросом")
def initialize():
    """Инициализация и обучение модели при первом запросе"""
    smart_home_ai.train_command_classifier()

if __name__ == '__main__':
    # Проверяем, существуют ли модели, если нет - обучаем
    if not os.path.exists(COMMAND_MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        smart_home_ai.train_command_classifier()
    
    # Запускаем сервер
    app.run(debug=True, host='0.0.0.0', port=5001)