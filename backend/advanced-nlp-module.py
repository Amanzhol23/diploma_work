# advanced_nlp.py - расширенный модуль обработки естественного языка для умного дома

import numpy as np
import re
import string
import os
import pickle
import json
import logging
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher

# Для работы с text-to-speech и speech-to-text
try:
    import pyttsx3
    import speech_recognition as sr

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 или speech_recognition не установлены. Функции TTS и STT недоступны.")

# Для обработки естественного языка
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer

    # Скачиваем необходимые модели NLTK, если их нет
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    NLP_ADVANCED = True
except ImportError:
    NLP_ADVANCED = False
    logging.warning("NLTK не установлен. Будет использована базовая обработка текста.")

# Для машинного обучения
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("scikit-learn не установлен. Функции машинного обучения недоступны.")

# Для продвинутой обработки естественного языка
try:
    import spacy

    # Загружаем модель для русского языка
    nlp_ru = spacy.load("ru_core_news_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spacy не установлен. Продвинутые функции NLP недоступны.")
except OSError:
    SPACY_AVAILABLE = False
    logging.warning(
        "Модель spacy для русского языка не установлена. Выполните: python -m spacy download ru_core_news_sm")

# Для нейронных сетей
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logging.warning("TensorFlow не установлен. Функции глубокого обучения недоступны.")

# Для работы с трансформерами
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers не установлен. Функции трансформеров недоступны.")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nlp_module.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Путь к директории с моделями и данными
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'nlp_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Путь к файлам моделей
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, 'intent_classifier.pkl')
NER_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_model.pkl')
LANGUAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'language_model.pkl')
COMMANDS_DATASET_PATH = os.path.join(MODEL_DIR, 'commands_dataset.json')
USER_PREFERENCES_PATH = os.path.join(MODEL_DIR, 'user_preferences.json')
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, 'transformer_model')
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, 'embeddings.pkl')


class SmartHomeNLP:
    """
    Класс для обработки естественного языка в системе умного дома
    с поддержкой русского и казахского языков
    """

    def __init__(self, language='ru'):
        """
        Инициализация класса для обработки естественного языка

        Args:
            language (str): Язык по умолчанию ('ru' или 'kk')
        """
        self.language = language
        self.stemmer_ru = SnowballStemmer("russian") if NLP_ADVANCED else None

        # Загрузка стоп-слов для русского языка
        if NLP_ADVANCED:
            self.stop_words_ru = set(stopwords.words('russian'))
        else:
            self.stop_words_ru = set(['и', 'в', 'на', 'с', 'по', 'для', 'от', 'к', 'у'])

        # Базовые стоп-слова для казахского языка
        self.stop_words_kk = set(['және', 'мен', 'бар', 'жоқ', 'үшін', 'бойынша'])

        # Инициализация модели интентов
        self.intent_classifier = None
        self.intent_vectorizer = None

        # Инициализация трансформерной модели
        self.transformer_model = None
        self.transformer_tokenizer = None

        # Нейросетевая модель для более сложного понимания языка
        self.nn_model = None
        self.tokenizer = None
        self.max_sequence_length = 50

        # Список поддерживаемых интентов
        self.intents = {
            'light_control': ['свет', 'освещение', 'лампа', 'люстра', 'жарық', 'шам'],
            'door_control': ['дверь', 'двери', 'есік', 'есіктер'],
            'window_control': ['окно', 'окна', 'терезе', 'терезелер'],
            'climate_control': ['температура', 'климат', 'кондиционер', 'температура', 'жылу'],
            'security': ['безопасность', 'сигнализация', 'камера', 'қауіпсіздік'],
            'media': ['музыка', 'телевизор', 'медиа', 'музыка', 'теледидар'],
            'query': ['сколько', 'какой', 'что', 'где', 'қанша', 'қандай', 'не', 'қайда'],
            'greeting': ['привет', 'здравствуй', 'доброе утро', 'сәлем', 'қайырлы таң'],
            'farewell': ['пока', 'до свидания', 'сау болыңыз', 'көріскенше'],
            'thanks': ['спасибо', 'благодарю', 'рахмет', 'алғыс'],
            'help': ['помоги', 'помощь', 'инструкция', 'көмектес', 'көмек']
        }

        # Словарь команд и соответствующих им действий для каждого языка
        self.commands = {
            'ru': {
                'light_on': ['включи свет', 'зажги свет', 'светло'],
                'light_off': ['выключи свет', 'погаси свет', 'темно'],
                'door_open': ['открой дверь', 'отвори дверь'],
                'door_close': ['закрой дверь', 'затвори дверь'],
                'window_open': ['открой окно', 'проветри'],
                'window_close': ['закрой окно'],
                'temperature_up': ['сделай теплее', 'увеличь температуру', 'жарко'],
                'temperature_down': ['сделай прохладнее', 'уменьши температуру', 'холодно'],
                'music_on': ['включи музыку', 'поставь музыку'],
                'music_off': ['выключи музыку', 'останови музыку'],
                'tv_on': ['включи телевизор', 'включи тв'],
                'tv_off': ['выключи телевизор', 'выключи тв'],
                'alarm_on': ['включи сигнализацию', 'охрана'],
                'alarm_off': ['выключи сигнализацию', 'сними с охраны'],
                'all_off': ['выключи всё', 'выключи все', 'всё выключи']
            },
            'kk': {
                'light_on': ['жарықты қос', 'шамды жақ'],
                'light_off': ['жарықты өшір', 'шамды сөндір'],
                'door_open': ['есікті аш'],
                'door_close': ['есікті жап'],
                'window_open': ['терезені аш', 'желдет'],
                'window_close': ['терезені жап'],
                'temperature_up': ['жылырақ жаса', 'температураны көтер'],
                'temperature_down': ['салқынырақ жаса', 'температураны төмендет'],
                'music_on': ['музыканы қос'],
                'music_off': ['музыканы өшір'],
                'tv_on': ['теледидарды қос'],
                'tv_off': ['теледидарды өшір'],
                'alarm_on': ['дабылды қос', 'күзетке қой'],
                'alarm_off': ['дабылды өшір', 'күзеттен шығар'],
                'all_off': ['бәрін өшір', 'барлығын өшір']
            }
        }

        # Словарь шаблонов команд для извлечения параметров
        self.command_patterns = {
            'ru': {
                r'(включи|зажги)\s+свет\s+в\s+(.+)': {'action': 'light_on', 'param': 'room'},
                r'(выключи|погаси)\s+свет\s+в\s+(.+)': {'action': 'light_off', 'param': 'room'},
                r'(открой|отвори)\s+(.+)\s+дверь': {'action': 'door_open', 'param': 'door_type'},
                r'(закрой|затвори)\s+(.+)\s+дверь': {'action': 'door_close', 'param': 'door_type'},
                r'(включи|зажги)\s+(.+)\s+свет': {'action': 'light_on', 'param': 'light_type'},
                r'(выключи|погаси)\s+(.+)\s+свет': {'action': 'light_off', 'param': 'light_type'},
                r'установи\s+температуру\s+(\d+)': {'action': 'set_temperature', 'param': 'value'},
                r'(включи|поставь)\s+(.+)\s+музыку': {'action': 'music_play', 'param': 'genre'},
                r'громкость\s+на\s+(\d+)': {'action': 'set_volume', 'param': 'value'}
            },
            'kk': {
                r'(жарықты қос|шамды жақ)\s+(.+)': {'action': 'light_on', 'param': 'room'},
                r'(жарықты өшір|шамды сөндір)\s+(.+)': {'action': 'light_off', 'param': 'room'},
                r'есікті аш\s+(.+)': {'action': 'door_open', 'param': 'door_type'},
                r'есікті жап\s+(.+)': {'action': 'door_close', 'param': 'door_type'}
            }
        }

        # Данные для обучения
        self.training_data = []

        # Словарь для хранения пользовательских предпочтений
        self.user_preferences = {}

        # История диалогов для контекстного анализа
        self.dialogue_history = {}

        # Загрузка моделей, если они существуют
        self.load_models()

        # Инициализация трансформерной модели при наличии зависимостей
        if TRANSFORMERS_AVAILABLE:
            self.init_transformer_model()

    def init_transformer_model(self):
        """Инициализация трансформерной модели"""
        try:
            if os.path.exists(TRANSFORMER_MODEL_PATH):
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_PATH)
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_PATH)
                logger.info("Трансформерная модель загружена из локального хранилища")
            else:
                # Используем предобученную модель на русском языке
                model_name = "cointegrated/rubert-tiny"
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)

                # Сохраняем модель локально
                self.transformer_tokenizer.save_pretrained(TRANSFORMER_MODEL_PATH)
                self.transformer_model.save_pretrained(TRANSFORMER_MODEL_PATH)
                logger.info(f"Трансформерная модель {model_name} загружена и сохранена локально")

            # Инициализация пайплайна для классификации текста
            self.text_classifier = pipeline(
                "text-classification",
                model=self.transformer_model,
                tokenizer=self.transformer_tokenizer
            )
            logger.info("Трансформерная модель инициализирована успешно")
        except Exception as e:
            logger.error(f"Ошибка при инициализации трансформерной модели: {str(e)}")
            self.transformer_model = None
            self.transformer_tokenizer = None

    def load_models(self):
        """Загрузка моделей из файлов"""
        try:
            # Загрузка модели интентов
            if os.path.exists(INTENT_MODEL_PATH):
                with open(INTENT_MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                    self.intent_classifier = model_data['classifier']
                    self.intent_vectorizer = model_data['vectorizer']
                logger.info("Модель классификации интентов загружена успешно")

            # Загрузка датасета команд
            if os.path.exists(COMMANDS_DATASET_PATH):
                with open(COMMANDS_DATASET_PATH, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                logger.info(f"Датасет команд загружен: {len(self.training_data)} записей")

            # Загрузка пользовательских предпочтений
            if os.path.exists(USER_PREFERENCES_PATH):
                with open(USER_PREFERENCES_PATH, 'r', encoding='utf-8') as f:
                    self.user_preferences = json.load(f)
                logger.info(f"Пользовательские предпочтения загружены: {len(self.user_preferences)} пользователей")

            # Загрузка нейросетевой модели
            if DL_AVAILABLE and os.path.exists(os.path.join(MODEL_DIR, 'nn_model.h5')):
                self.nn_model = load_model(os.path.join(MODEL_DIR, 'nn_model.h5'))
                # Загрузка токенизатора
                if os.path.exists(os.path.join(MODEL_DIR, 'tokenizer.json')):
                    with open(os.path.join(MODEL_DIR, 'tokenizer.json'), 'r', encoding='utf-8') as f:
                        tokenizer_json = json.load(f)
                        self.tokenizer = Tokenizer()
                        self.tokenizer.word_index = tokenizer_json
                logger.info("Нейросетевая модель загружена успешно")

        except Exception as e:
            logger.error(f"Ошибка при загрузке моделей: {str(e)}")
            # Если возникла ошибка, модели будут созданы заново при обучении

    def save_models(self):
        """Сохранение моделей в файлы"""
        try:
            # Сохранение модели интентов
            if self.intent_classifier and self.intent_vectorizer:
                model_data = {
                    'classifier': self.intent_classifier,
                    'vectorizer': self.intent_vectorizer
                }
                with open(INTENT_MODEL_PATH, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info("Модель классификации интентов сохранена")

            # Сохранение датасета команд
            with open(COMMANDS_DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Датасет команд сохранен: {len(self.training_data)} записей")

            # Сохранение пользовательских предпочтений
            with open(USER_PREFERENCES_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, ensure_ascii=False, indent=2)
            logger.info(f"Пользовательские предпочтения сохранены")

            # Сохранение нейросетевой модели
            if DL_AVAILABLE and self.nn_model:
                self.nn_model.save(os.path.join(MODEL_DIR, 'nn_model.h5'))
                # Сохранение токенизатора
                if self.tokenizer:
                    with open(os.path.join(MODEL_DIR, 'tokenizer.json'), 'w', encoding='utf-8') as f:
                        json.dump(self.tokenizer.word_index, f, ensure_ascii=False)
                logger.info("Нейросетевая модель сохранена")

        except Exception as e:
            logger.error(f"Ошибка при сохранении моделей: {str(e)}")

    def preprocess_text(self, text, lang='auto'):
        """
        Предобработка текста: нормализация, токенизация, удаление стоп-слов, стемминг

        Args:
            text (str): Текст для обработки
            lang (str): Язык текста ('ru', 'kk' или 'auto' для автоопределения)

        Returns:
            str: Обработанный текст
        """
        if not text:
            return ""

        # Если язык автоматический, определяем его
        if lang == 'auto':
            lang = self.detect_language(text)

        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление пунктуации
        text = re.sub(f'[{string.punctuation}]', ' ', text)

        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        # Если доступны продвинутые NLP функции
        if NLP_ADVANCED:
            # Токенизация
            tokens = word_tokenize(text, language='russian' if lang == 'ru' else 'english')

            # Удаление стоп-слов
            stop_words = self.stop_words_ru if lang == 'ru' else self.stop_words_kk
            tokens = [token for token in tokens if token not in stop_words]

            # Стемминг для русского языка
            if lang == 'ru' and self.stemmer_ru:
                tokens = [self.stemmer_ru.stem(token) for token in tokens]

            # Соединение токенов обратно в текст
            return ' '.join(tokens)
        else:
            # Базовая обработка без NLTK
            return text

    def detect_language(self, text):
        """
        Определение языка текста

        Args:
            text (str): Текст для определения языка

        Returns:
            str: Код языка ('ru', 'kk' или 'unknown')
        """
        # Простой детектор языка по характерным буквам
        text = text.lower()

        # Характерные буквы русского алфавита
        russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

        # Характерные буквы казахского алфавита, которых нет в русском
        kazakh_chars = set('әғқңөұүһ')

        # Подсчет букв
        russian_count = sum(1 for char in text if char in russian_chars)
        kazakh_count = sum(1 for char in text if char in kazakh_chars)

        # Если есть казахские буквы, считаем текст казахским
        if kazakh_count > 0:
            return 'kk'
        # Если есть русские буквы, считаем текст русским
        elif russian_count > 0:
            return 'ru'
        # Иначе не можем определить
        else:
            return 'unknown'

    def train_intent_classifier(self):
        """
        Обучение классификатора интентов на основе датасета
        """
        if not ML_AVAILABLE:
            logger.warning("Невозможно обучить классификатор - не установлен scikit-learn")
            return False

        try:
            # Подготовка данных для обучения
            X = []  # Тексты команд
            y = []  # Соответствующие им интенты

            # Добавление примеров из предопределенных команд
            for intent, examples in self.intents.items():
                for example in examples:
                    X.append(example)
                    y.append(intent)

            # Добавление примеров из датасета обучения
            for entry in self.training_data:
                if 'text' in entry and 'intent' in entry:
                    X.append(entry['text'])
                    y.append(entry['intent'])

            if len(X) < 10:
                logger.warning("Недостаточно данных для обучения классификатора")
                return False

            # Предобработка текстов
            X_processed = [self.preprocess_text(text) for text in X]

            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )

            # Создание векторизатора
            self.intent_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),  # Униграммы и биграммы
                max_features=5000,  # Максимальное количество признаков
                min_df=2  # Минимальная частота документов
            )

            # Векторизация текстов
            X_train_vec = self.intent_vectorizer.fit_transform(X_train)
            X_test_vec = self.intent_vectorizer.transform(X_test)

            # Создание классификатора
            self.intent_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )

            # Обучение классификатора
            self.intent_classifier.fit(X_train_vec, y_train)

            # Оценка качества
            y_pred = self.intent_classifier.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Классификатор интентов обучен с точностью: {accuracy:.2f}")

            # Сохранение моделей
            self.save_models()

            return True
        except Exception as e:
            logger.error(f"Ошибка при обучении классификатора интентов: {str(e)}")
            return False

    def train_neural_network(self):
        """
        Обучение нейронной сети для классификации команд
        """
        if not DL_AVAILABLE:
            logger.warning("Невозможно обучить нейронную сеть - не установлен TensorFlow")
            return False

        try:
            # Подготовка данных для обучения
            texts = []
            labels = []

            # Создание датасета из команд
            for lang, commands_dict in self.commands.items():
                for command_class, examples in commands_dict.items():
                    for example in examples:
                        texts.append(example)
                        labels.append(command_class)

            # Добавление данных из обучающего датасета
            for entry in self.training_data:
                if 'text' in entry and 'command' in entry:
                    texts.append(entry['text'])
                    labels.append(entry['command'])

            if len(texts) < 10:
                logger.warning("Недостаточно данных для обучения нейронной сети")
                return False

            # Преобразование меток в числовой формат
            label_dict = {label: i for i, label in enumerate(set(labels))}
            num_labels = len(label_dict)
            numeric_labels = [label_dict[label] for label in labels]

            # Токенизация текстов
            self.tokenizer = Tokenizer(num_words=10000)
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)

            # Определение максимальной длины последовательности
            max_length = max(len(seq) for seq in sequences)
            self.max_sequence_length = min(self.max_sequence_length, max_length)

            # Предобработка последовательностей
            data = pad_sequences(sequences, maxlen=self.max_sequence_length)

            # Преобразование меток в категориальный формат
            labels_categorical = tf.keras.utils.to_categorical(numeric_labels, num_classes=num_labels)

            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                data, labels_categorical, test_size=0.2, random_state=42
            )

            # Создание модели
            vocab_size = len(self.tokenizer.word_index) + 1
            embedding_dim = 100

            self.nn_model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=self.max_sequence_length),
                Bidirectional(LSTM(64, return_sequences=True)),
                Bidirectional(LSTM(32)),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(num_labels, activation='softmax')
            ])

            # Компиляция модели
            self.nn_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Обучение модели
            history = self.nn_model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=1
            )

            # Оценка качества
            loss, accuracy = self.nn_model.evaluate(X_test, y_test)
            logger.info(f"Нейронная сеть обучена с точностью: {accuracy:.2f}")

            # Сохранение модели и меток
            self.nn_model.save(os.path.join(MODEL_DIR, 'nn_model.h5'))
            with open(os.path.join(MODEL_DIR, 'label_dict.json'), 'w', encoding='utf-8') as f:
                json.dump({v: k for k, v in label_dict.items()}, f, ensure_ascii=False)

            # Сохранение токенизатора
            with open(os.path.join(MODEL_DIR, 'tokenizer.json'), 'w', encoding='utf-8') as f:
                json.dump(self.tokenizer.word_index, f, ensure_ascii=False)

            logger.info("Обучение нейронной сети завершено успешно")
            return True

        except Exception as e:
            logger.error(f"Ошибка при обучении нейронной сети: {str(e)}")
            return False

    def predict_intent(self, text, user_id=None):
        """
        Предсказание интента команды

        Args:
            text (str): Введенная команда
            user_id (str, optional): Идентификатор пользователя для учета предпочтений

        Returns:
            dict: Результат с предсказанным интентом и уверенностью
        """
        if not text:
            return {'intent': 'unknown', 'confidence': 0.0}

        lang = self.detect_language(text)
        processed_text = self.preprocess_text(text, lang)

        if not self.intent_classifier or not self.intent_vectorizer:
            self.train_intent_classifier()

        try:
            if self.intent_classifier and self.intent_vectorizer:
                # Векторизация текста
                text_vec = self.intent_vectorizer.transform([processed_text])
                # Предсказание
                intent = self.intent_classifier.predict(text_vec)[0]
                confidence = self.intent_classifier.predict_proba(text_vec)[0].max()

                # Учет пользовательских предпочтений
                if user_id and user_id in self.user_preferences:
                    user_prefs = self.user_preferences[user_id].get('intents', {})
                    if intent in user_prefs:
                        confidence *= 1.1  # Увеличиваем уверенность для частых интентов

                logger.info(f"Предсказан интент: {intent} с уверенностью {confidence:.2f}")
                return {'intent': intent, 'confidence': confidence}
            else:
                logger.warning("Классификатор интентов не обучен")
                return {'intent': 'unknown', 'confidence': 0.0}
        except Exception as e:
            logger.error(f"Ошибка при предсказании интента: {str(e)}")
            return {'intent': 'unknown', 'confidence': 0.0}

    def predict_command(self, text, user_id=None):
        """
        Предсказание команды с использованием нейронной сети

        Args:
            text (str): Введенная команда
            user_id (str, optional): Идентификатор пользователя

        Returns:
            dict: Результат с предсказанной командой и уверенностью
        """
        if not DL_AVAILABLE or not self.nn_model or not self.tokenizer:
            logger.warning("Нейронная сеть недоступна или не обучена")
            return {'command': 'unknown', 'confidence': 0.0}

        try:
            # Предобработка текста
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)

            # Предсказание
            prediction = self.nn_model.predict(padded_sequence)[0]
            confidence = np.max(prediction)
            command_index = np.argmax(prediction)

            # Загрузка словаря меток
            with open(os.path.join(MODEL_DIR, 'label_dict.json'), 'r', encoding='utf-8') as f:
                label_dict = json.load(f)

            command = label_dict[str(command_index)]

            # Учет пользовательских предпочтений
            if user_id and user_id in self.user_preferences:
                user_prefs = self.user_preferences[user_id].get('commands', {})
                if command in user_prefs:
                    confidence *= 1.1  # Увеличиваем уверенность для частых команд

            logger.info(f"Предсказана команда: {command} с уверенностью {confidence:.2f}")
            return {'command': command, 'confidence': float(confidence)}
        except Exception as e:
            logger.error(f"Ошибка при предсказании команды: {str(e)}")
            return {'command': 'unknown', 'confidence': 0.0}

    def extract_parameters(self, text):
        """
        Извлечение параметров из команды с использованием шаблонов

        Args:
            text (str): Введенная команда

        Returns:
            dict: Словарь с действием и параметрами
        """
        lang = self.detect_language(text)
        patterns = self.command_patterns.get(lang, {})

        for pattern, action_info in patterns.items():
            match = re.match(pattern, text.lower())
            if match:
                param_value = match.group(2) if len(match.groups()) >= 2 else match.group(1)
                return {
                    'action': action_info['action'],
                    'param': {action_info['param']: param_value}
                }

        return {'action': None, 'param': {}}

    def process_command(self, text, user_id='anonymous'):
        """
        Полная обработка команды с учетом интента, команды и параметров

        Args:
            text (str): Введенная команда
            user_id (str): Идентификатор пользователя

        Returns:
            dict: Результат обработки команды
        """
        intent_result = self.predict_intent(text, user_id)
        command_result = self.predict_command(text, user_id)
        params = self.extract_parameters(text)

        # Логика объединения результатов
        if command_result['confidence'] > 0.7:
            action = command_result['command']
            confidence = command_result['confidence']
        elif intent_result['confidence'] > 0.5:
            action = intent_result['intent']
            confidence = intent_result['confidence']
        else:
            action = 'unknown'
            confidence = max(intent_result['confidence'], command_result['confidence'])

        # Сохранение в историю диалогов
        if user_id not in self.dialogue_history:
            self.dialogue_history[user_id] = []
        self.dialogue_history[user_id].append({
            'text': text,
            'intent': intent_result['intent'],
            'command': command_result['command'],
            'params': params,
            'timestamp': datetime.now().isoformat()
        })

        # Обновление пользовательских предпочтений
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {'intents': {}, 'commands': {}}

        self.user_preferences[user_id]['intents'][intent_result['intent']] = \
            self.user_preferences[user_id]['intents'].get(intent_result['intent'], 0) + 1
        self.user_preferences[user_id]['commands'][command_result['command']] = \
            self.user_preferences[user_id]['commands'].get(command_result['command'], 0) + 1

        self.save_models()

        return {
            'action': action if params['action'] is None else params['action'],
            'intent': intent_result['intent'],
            'command': command_result['command'],
            'confidence': confidence,
            'params': params['param'],
            'language': self.detect_language(text)
        }

    def speech_to_text(self):
        """
        Преобразование речи в текст

        Returns:
            str: Распознанный текст или None при ошибке
        """
        if not TTS_AVAILABLE:
            logger.warning("Speech Recognition не доступен")
            return None

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            logger.info("Слушаю...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio, language='ru-RU' if self.language == 'ru' else 'kk-KZ')
                logger.info(f"Распознан текст: {text}")
                return text
            except sr.UnknownValueError:
                logger.error("Не удалось распознать речь")
                return None
            except sr.RequestError as e:
                logger.error(f"Ошибка сервиса распознавания речи: {str(e)}")
                return None

    def text_to_speech(self, text):
        """
        Преобразование текста в речь

        Args:
            text (str): Текст для озвучивания
        """
        if not TTS_AVAILABLE:
            logger.warning("Text-to-Speech не доступен")
            return

        engine = pyttsx3.init()
        engine.setProperty('voice', 'ru' if self.language == 'ru' else 'kk')
        engine.say(text)
        engine.runAndWait()
        logger.info(f"Озвучен текст: {text}")


if __name__ == "__main__":
    nlp = SmartHomeNLP()
    # Пример использования
    command = "включи свет в спальне"
    result = nlp.process_command(command, user_id="user1")
    print(result)

    # Обучение моделей
    nlp.train_intent_classifier()
    nlp.train_neural_network()