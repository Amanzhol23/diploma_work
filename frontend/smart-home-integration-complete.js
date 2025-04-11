// Файл smart-home-integration.js - включить его в ваш house.html

// Адрес API искусственного интеллекта
const AI_API_URL = 'http://localhost:5001/api';

// Функция для обработки команды через AI
async function processCommandWithAI(command, username) {
    try {
        // Отправляем команду на AI сервер для обработки и классификации
        const response = await fetch(`${AI_API_URL}/process-command`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                command: command,
                user_id: username
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("AI обработал команду:", result);
        
        // Если AI уверен в команде с вероятностью > 0.7
        if (result.confidence > 0.7) {
            // Здесь можно добавить дополнительную логику обработки команды
            return result;
        } else {
            // Если уверенность низкая, используем стандартную обработку
            console.log("Низкая уверенность AI, используем стандартную обработку");
            return { useDefault: true };
        }
    } catch (error) {
        console.error("Ошибка при обработке команды через AI:", error);
        // В случае ошибки используем стандартную обработку
        return { useDefault: true };
    }
}
function setupVoiceRecognition() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    let currentLang = 'ru-RU';
    recognition.lang = currentLang;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        console.log('Распознанный текст:', transcript);
        document.getElementById('command').value = transcript;
        executeCommand(); // Автоматический вызов
    };

    recognition.onerror = (event) => {
        console.error('Ошибка распознавания:', event.error);
        document.getElementById('status').textContent = 'Ошибка распознавания речи';
    };

    recognition.onend = () => {
        document.getElementById('voiceBtn').classList.remove('active');
    };

    return {
        recognition,
        start: () => {
            console.log('Начинаем распознавание...');
            recognition.start();
        },
        switchLanguage: () => {
            currentLang = currentLang === 'ru-RU' ? 'kk-KZ' : 'ru-RU';
            recognition.lang = currentLang;
            document.getElementById('switchLangBtn').textContent = currentLang === 'ru-RU' ? 'RU/KZ' : 'KZ/RU';
            console.log('Язык изменен на:', currentLang);
        }
    };
}
window.aiHelpers.setupVoiceRecognition = setupVoiceRecognition;
// Функция для получения предложений от AI
async function getAISuggestions(username) {
    try {
        const response = await fetch(`${AI_API_URL}/suggestions?user_id=${username}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result.suggestions || [];
    } catch (error) {
        console.error("Ошибка при получении предложений от AI:", error);
        return [];
    }
}

// Функция для отображения предложений от AI
function displayAISuggestions(suggestions) {
    // Проверяем, существует ли контейнер для предложений
    let suggestionsContainer = document.getElementById('ai-suggestions');
    
    if (!suggestionsContainer) {
        // Если контейнер не существует, создаем его
        suggestionsContainer = document.createElement('div');
        suggestionsContainer.id = 'ai-suggestions';
        suggestionsContainer.style.marginTop = '10px';
        suggestionsContainer.style.padding = '10px';
        suggestionsContainer.style.backgroundColor = '#f0f0f0';
        suggestionsContainer.style.borderRadius = '5px';
        
        // Добавляем заголовок
        const header = document.createElement('h3');
        header.textContent = 'Предложения AI:';
        suggestionsContainer.appendChild(header);
        
        // Создаем список для предложений
        const suggestionsList = document.createElement('ul');
        suggestionsList.id = 'suggestions-list';
        suggestionsContainer.appendChild(suggestionsList);
        
        // Находим место для вставки контейнера (после элемента status)
        const statusElement = document.getElementById('status');
        statusElement.parentNode.insertBefore(suggestionsContainer, statusElement.nextSibling);
    }
    
    // Очищаем и обновляем список предложений
    const suggestionsList = document.getElementById('suggestions-list');
    suggestionsList.innerHTML = '';
    
    if (suggestions.length === 0) {
        const noSuggestions = document.createElement('li');
        noSuggestions.textContent = 'Нет предложений';
        suggestionsList.appendChild(noSuggestions);
    } else {
        suggestions.forEach(suggestion => {
            const suggestionItem = document.createElement('li');
            suggestionItem.textContent = suggestion;
            suggestionItem.style.cursor = 'pointer';
            suggestionItem.style.padding = '5px';
            suggestionItem.style.marginBottom = '5px';
            suggestionItem.style.backgroundColor = '#e0e0e0';
            suggestionItem.style.borderRadius = '3px';
            
            // Добавляем обработчик клика для выполнения предложения
            suggestionItem.addEventListener('click', function() {
                document.getElementById('command').value = suggestion;
                executeCommand(); // Предполагается, что эта функция существует в house.html
            });
            
            suggestionsList.appendChild(suggestionItem);
        });
    }
}

// Функция для улучшенного распознавания голоса с поддержкой двух языков
function setupImprovedVoiceRecognition() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    
    // Настройка для мультиязычности (русский и казахский)
    const supportedLanguages = ['ru-RU', 'kk-KZ'];
    let currentLangIndex = 0;
    recognition.lang = supportedLanguages[currentLangIndex];
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 3;
    
    // Улучшенная обработка результатов с выбором наиболее вероятного варианта
    recognition.onresult = function(event) {
        const results = event.results[0];
        let transcriptions = [];
        
        // Собираем все альтернативы
        for (let i = 0; i < results.length; i++) {
            transcriptions.push({
                text: results[i].transcript,
                confidence: results[i].confidence
            });
        }
        
        // Сортируем по уверенности
        transcriptions.sort((a, b) => b.confidence - a.confidence);
        
        // Берем лучший результат
        const bestTranscription = transcriptions[0].text;
        console.log("Распознанная команда:", bestTranscription);
        console.log("Альтернативы:", transcriptions);
        
        // Устанавливаем распознанную команду в поле ввода
        const commandInput = document.getElementById('command');
        if (commandInput) {
            commandInput.value = bestTranscription;
            
            // Автоматически выполняем команду, если уверенность высокая
            if (transcriptions[0].confidence > 0.8) {
                executeCommand(); // Предполагается, что эта функция существует в house.html
            }
        }
    };
    
    // Обработка ошибок
    recognition.onerror = function(event) {
        console.error("Ошибка распознавания речи:", event.error);
        
        // Если ошибка связана с языком, пробуем другой язык
        if (event.error === 'language-not-supported' || event.error === 'no-speech') {
            currentLangIndex = (currentLangIndex + 1) % supportedLanguages.length;
            recognition.lang = supportedLanguages[currentLangIndex];
            recognition.start();
        }
        
        // Сообщаем пользователю об ошибке
        updateStatus("Ошибка распознавания речи. Пожалуйста, попробуйте еще раз.");
    };
    
    recognition.onend = function() {
        console.log("Распознавание речи завершено");
        updateStatus("Готов к приему команд");
    };
    
    // Функция для обновления статуса (предполагается, что она существует)
    function updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }
    
    // Функция для начала распознавания
    function startListening() {
        try {
            updateStatus("Слушаю команду...");
            recognition.start();
        } catch (error) {
            console.error("Ошибка при запуске распознавания:", error);
            updateStatus("Не удалось запустить распознавание речи.");
        }
    }
    
    // Функция для переключения языка
    function switchLanguage() {
        currentLangIndex = (currentLangIndex + 1) % supportedLanguages.length;
        recognition.lang = supportedLanguages[currentLangIndex];
        console.log("Язык распознавания переключен на:", recognition.lang);
        
        // Показываем текущий язык
        const langNames = {
            'ru-RU': 'Русский',
            'kk-KZ': 'Қазақша'
        };
        updateStatus(`Язык распознавания: ${langNames[recognition.lang]}`);
    }
    
    // Возвращаем объект с функциями для управления распознаванием
    return {
        start: startListening,
        switchLanguage: switchLanguage,
        recognition: recognition
    };
}

// Функция для периодического обновления предложений AI
function setupAISuggestionUpdater(username, interval = 60000) {
    // Первоначальное получение предложений
    updateSuggestions();
    
    // Устанавливаем интервал для обновления
    const updaterInterval = setInterval(updateSuggestions, interval);
    
    async function updateSuggestions() {
        const suggestions = await getAISuggestions(username);
        displayAISuggestions(suggestions);
    }
    
    // Возвращаем функцию для отмены интервала
    return function stopUpdater() {
        clearInterval(updaterInterval);
    };
}

// Функция для получения аналитики из AI
async function getAIAnalytics(username = null) {
    try {
        let url = `${AI_API_URL}/analytics`;
        if (username) {
            url += `?user_id=${username}`;
        }
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error("Ошибка при получении аналитики:", error);
        return null;
    }
}

// Функция для отображения аналитики
function displayAnalytics(analytics) {
    if (!analytics) {
        console.error("Нет данных для отображения аналитики");
        return;
    }
    
    // Создаем/находим контейнер для аналитики
    let analyticsContainer = document.getElementById('ai-analytics');
    if (!analyticsContainer) {
        analyticsContainer = document.createElement('div');
        analyticsContainer.id = 'ai-analytics';
        analyticsContainer.style.marginTop = '20px';
        analyticsContainer.style.padding = '15px';
        analyticsContainer.style.backgroundColor = '#f5f5f5';
        analyticsContainer.style.borderRadius = '5px';
        analyticsContainer.style.border = '1px solid #ddd';
        
        // Добавляем заголовок
        const header = document.createElement('h3');
        header.textContent = 'Аналитика использования';
        analyticsContainer.appendChild(header);
        
        // Находим место для вставки контейнера
        const container = document.querySelector('.container');
        if (container) {
            container.appendChild(analyticsContainer);
        } else {
            document.body.appendChild(analyticsContainer);
        }
    }
    
    // Очищаем контейнер
    analyticsContainer.innerHTML = '<h3>Аналитика использования</h3>';
    
    // Добавляем общую информацию
    const totalCommands = document.createElement('p');
    totalCommands.innerHTML = `<strong>Всего команд:</strong> ${analytics.total_commands}`;
    analyticsContainer.appendChild(totalCommands);
    
    // Создаем раздел для статистики команд
    const commandStatsSection = document.createElement('div');
    commandStatsSection.innerHTML = '<h4>Статистика команд</h4>';
    
    // Преобразуем статистику команд в массив для сортировки
    const commandStatsArray = Object.entries(analytics.command_stats || {})
        .map(([command, count]) => ({ command, count }))
        .sort((a, b) => b.count - a.count);
    
    if (commandStatsArray.length > 0) {
        const commandsList = document.createElement('ul');
        commandStatsArray.forEach(({ command, count }) => {
            const item = document.createElement('li');
            item.textContent = `${getCommandName(command)}: ${count}`;
            commandsList.appendChild(item);
        });
        commandStatsSection.appendChild(commandsList);
    } else {
        commandStatsSection.innerHTML += '<p>Нет данных о командах</p>';
    }
    
    analyticsContainer.appendChild(commandStatsSection);
    
    // Создаем раздел для статистики комнат
    if (analytics.room_stats) {
        const roomStatsSection = document.createElement('div');
        roomStatsSection.innerHTML = '<h4>Статистика комнат</h4>';
        
        const roomStatsArray = Object.entries(analytics.room_stats)
            .map(([room, count]) => ({ room, count }))
            .sort((a, b) => b.count - a.count);
        
        if (roomStatsArray.length > 0) {
            const roomsList = document.createElement('ul');
            roomStatsArray.forEach(({ room, count }) => {
                const item = document.createElement('li');
                item.textContent = `${getRoomName(room)}: ${count}`;
                roomsList.appendChild(item);
            });
            roomStatsSection.appendChild(roomsList);
        } else {
            roomStatsSection.innerHTML += '<p>Нет данных о комнатах</p>';
        }
        
        analyticsContainer.appendChild(roomStatsSection);
    }
    
    // Добавляем временные паттерны, если они есть
    if (analytics.time_patterns) {
        const timePatternSection = document.createElement('div');
        timePatternSection.innerHTML = '<h4>Активность по времени суток</h4>';
        
        const timePatterns = [
            { label: 'Утро (5-12)', value: analytics.time_patterns.morning || 0 },
            { label: 'День (12-17)', value: analytics.time_patterns.afternoon || 0 },
            { label: 'Вечер (17-22)', value: analytics.time_patterns.evening || 0 },
            { label: 'Ночь (22-5)', value: analytics.time_patterns.night || 0 }
        ];
        
        const timeList = document.createElement('ul');
        timePatterns.forEach(({ label, value }) => {
            const item = document.createElement('li');
            item.textContent = `${label}: ${value}`;
            timeList.appendChild(item);
        });
        
        timePatternSection.appendChild(timeList);
        analyticsContainer.appendChild(timePatternSection);
    }
}

// Вспомогательная функция для получения читаемого названия команды
function getCommandName(commandClass) {
    const commandNames = {
        'light_on': 'Включение света',
        'light_off': 'Выключение света',
        'door_open': 'Открытие двери',
        'door_close': 'Закрытие двери',
        'window_open': 'Открытие окна',
        'window_close': 'Закрытие окна',
        'garage_open': 'Открытие гаража',
        'garage_close': 'Закрытие гаража',
        'roof_open': 'Открытие крыши',
        'roof_close': 'Закрытие крыши',
        'sprinkler_on': 'Включение полива',
        'sprinkler_off': 'Выключение полива',
        'all_lights_on': 'Включение всего света',
        'all_lights_off': 'Выключение всего света',
        'reset': 'Сброс'
    };
    
    return commandNames[commandClass] || commandClass;
}

// Вспомогательная функция для получения читаемого названия комнаты
function getRoomName(roomKey) {
    const roomNames = {
        'bedroom': 'Спальня',
        'kitchen': 'Кухня',
        'living_room': 'Гостиная',
        'hall': 'Холл',
        'entrance': 'Прихожая',
        'corridor': 'Коридор'
    };
    
    return roomNames[roomKey] || roomKey;
}

// Экспортируем функции для использования в house.html
window.aiHelpers = {
    processCommand: processCommandWithAI,
    getSuggestions: getAISuggestions,
    displaySuggestions: displayAISuggestions,
    setupVoiceRecognition: setupImprovedVoiceRecognition,
    setupSuggestionUpdater: setupAISuggestionUpdater,
    getAnalytics: getAIAnalytics,
    displayAnalytics: displayAnalytics
};
