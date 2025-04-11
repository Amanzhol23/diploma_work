document.addEventListener('DOMContentLoaded', () => {
    const username = 'user1';
    const voiceRecognition = window.aiHelpers.setupVoiceRecognition();
    let stopUpdater;
    let scene, camera, renderer, house, roof, garageDoor, sprinklers;

    initialize();

    function initialize() {
        setup3DScene();
        stopUpdater = window.aiHelpers.setupSuggestionUpdater(username);
        updateAnalytics();

        document.getElementById('executeBtn').addEventListener('click', executeCommand);
        document.getElementById('voiceBtn').addEventListener('click', () => {
            document.getElementById('voiceBtn').classList.add('active');
            voiceRecognition.start();
        });
        document.getElementById('switchLangBtn').addEventListener('click', () => {
            voiceRecognition.switchLanguage();
        });

        voiceRecognition.recognition.onend = () => {
            document.getElementById('voiceBtn').classList.remove('active');
            updateStatus("Готов к приему команд");
        };
    }

    function setup3DScene() {
        const container = document.getElementById('scene-container');
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / 400, 0.1, 1000);
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, 400);
        container.appendChild(renderer.domElement);

        // Освещение
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        scene.add(directionalLight);

        // Земля (газон)
        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.MeshBasicMaterial({ color: 0x228B22 });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        scene.add(ground);

        // Дом
        const houseGeometry = new THREE.BoxGeometry(8, 4, 6);
        const houseMaterial = new THREE.MeshPhongMaterial({ color: 0xD3D3D3 });
        house = new THREE.Mesh(houseGeometry, houseMaterial);
        house.position.y = 2;
        scene.add(house);

        // Крыша
        const roofGeometry = new THREE.BoxGeometry(8.2, 0.5, 6.2);
        const roofMaterial = new THREE.MeshPhongMaterial({ color: 0x8B4513 });
        roof = new THREE.Mesh(roofGeometry, roofMaterial);
        roof.position.set(0, 4.25, 0);
        scene.add(roof);

        // Гараж
        const garageGeometry = new THREE.BoxGeometry(4, 3, 4);
        const garageMaterial = new THREE.MeshPhongMaterial({ color: 0xA9A9A9 });
        const garage = new THREE.Mesh(garageGeometry, garageMaterial);
        garage.position.set(6, 1.5, 0);
        scene.add(garage);

        // Ворота гаража
        const doorGeometry = new THREE.PlaneGeometry(3, 2.5);
        const doorMaterial = new THREE.MeshPhongMaterial({ color: 0x696969 });
        garageDoor = new THREE.Mesh(doorGeometry, doorMaterial);
        garageDoor.position.set(6, 1.25, 2.01);
        scene.add(garageDoor);

        // Спринклеры (полив)
        sprinklers = [];
        for (let i = 0; i < 4; i++) {
            const sprinklerGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.5);
            const sprinklerMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
            const sprinkler = new THREE.Mesh(sprinklerGeometry, sprinklerMaterial);
            sprinkler.position.set(-5 + i * 3, 0.25, -5);
            scene.add(sprinkler);
            sprinklers.push(sprinkler);
        }

        camera.position.set(0, 5, 10);
        camera.lookAt(0, 2, 0);

        animate();
    }

    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }

    async function executeCommand() {
    const commandInput = document.getElementById('command');
    const command = commandInput.value.trim().toLowerCase();

    if (!command) {
        updateStatus('Пожалуйста, введите или произнесите команду');
        console.log('Команда пустая');
        return;
    }

    console.log('Отправка команды:', command);
    updateStatus('Обработка команды...');
    const result = await window.aiHelpers.processCommand(command, username);
    console.log('Ответ от backend:', result);

    if (!result || result.useDefault || result.confidence < 0.7) {
        updateStatus('Команда не распознана точно');
        console.log('Команда не распознана или уверенность низкая');
    } else {
        updateStatus(`Команда: ${result.command_class}, Уверенность: ${(result.confidence * 100).toFixed(2)}%, Комната: ${result.room || 'не указана'}`);
        console.log('Выполнение команды:', result.command_class);
        handleCommand(result.command_class, result.room);
    }

    commandInput.value = '';
    updateAnalytics();
}

    function handleCommand(commandClass, room) {
        if (commandClass === 'roof_open' || command === 'старт' || command === 'шатырды аш') {
            roof.position.y = 6;
            document.getElementById('roof-status').textContent = 'Открыта';
        } else if (commandClass === 'roof_close' || command === 'стоп' || command === 'шатырды жап') {
            roof.position.y = 4.25;
            document.getElementById('roof-status').textContent = 'Закрыта';
        } else if (commandClass === 'garage_open') {
            garageDoor.position.y = 3.75;
            document.getElementById('garage-status').textContent = 'Открыт';
        } else if (commandClass === 'garage_close') {
            garageDoor.position.y = 1.25;
            document.getElementById('garage-status').textContent = 'Закрыт';
        } else if (commandClass === 'sprinkler_on' || command === 'включи полив' || command === 'суаруды қос') {
            sprinklers.forEach(s => s.position.y = 0.5);
            document.getElementById('sprinkler-status').textContent = 'Включен';
        } else if (commandClass === 'sprinkler_off' || command === 'выключи полив' || command === 'суаруды өшір') {
            sprinklers.forEach(s => s.position.y = 0.25);
            document.getElementById('sprinkler-status').textContent = 'Выключен';
        } else if (commandClass === 'light_on') {
            document.getElementById('light-status').textContent = 'Включен';
        } else if (commandClass === 'light_off') {
            document.getElementById('light-status').textContent = 'Выключен';
        } else if (commandClass === 'door_open') {
            document.getElementById('door-status').textContent = 'Открыта';
        } else if (commandClass === 'door_close') {
            document.getElementById('door-status').textContent = 'Закрыта';
        } else if (commandClass === 'window_open') {
            document.getElementById('window-status').textContent = 'Открыто';
        } else if (commandClass === 'window_close') {
            document.getElementById('window-status').textContent = 'Закрыто';
        }
    }

    async function updateAnalytics() {
        const analytics = await window.aiHelpers.getAnalytics(username);
        window.aiHelpers.displayAnalytics(analytics);
    }

    function updateStatus(message) {
        const statusBox = document.getElementById('status');
        statusBox.textContent = message;
        statusBox.classList.add('fade');
        setTimeout(() => statusBox.classList.remove('fade'), 500);
    }
});