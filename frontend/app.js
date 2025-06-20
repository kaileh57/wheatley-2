// Wheatley 2.0 - Web Interface

class WheatleyClient {
    constructor() {
        this.ws = null;
        this.apiKey = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadApiKey();
    }
    
    initializeElements() {
        // Connection elements
        this.apiKeyInput = document.getElementById('apiKeyInput');
        this.connectButton = document.getElementById('connectButton');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.statusText = this.connectionStatus.querySelector('.status-text');
        
        // Chat elements
        this.messagesContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.voiceButton = document.getElementById('voiceButton');
        
        // Side panel elements
        this.activeTasksContainer = document.getElementById('activeTasks');
        this.currentActivity = document.getElementById('currentActivity');
        this.currentLocation = document.getElementById('currentLocation');
        this.currentMood = document.getElementById('currentMood');
        
        // Audio visualizer
        this.audioVisualizer = document.getElementById('audioVisualizer');
    }
    
    attachEventListeners() {
        // Connection
        this.connectButton.addEventListener('click', () => this.toggleConnection());
        this.apiKeyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.toggleConnection();
        });
        
        // Chat
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Voice
        this.voiceButton.addEventListener('click', () => this.toggleVoiceRecording());
        
        // Quick actions
        document.querySelectorAll('.action-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                this.handleQuickAction(action);
            });
        });
    }
    
    loadApiKey() {
        const savedKey = localStorage.getItem('wheatley_api_key');
        if (savedKey) {
            this.apiKeyInput.value = savedKey;
        }
    }
    
    saveApiKey() {
        if (this.apiKey) {
            localStorage.setItem('wheatley_api_key', this.apiKey);
        }
    }
    
    async toggleConnection() {
        if (this.isConnected) {
            this.disconnect();
        } else {
            await this.connect();
        }
    }
    
    async connect() {
        const apiKey = this.apiKeyInput.value.trim();
        if (!apiKey) {
            this.showMessage('Please enter your API key', 'error');
            return;
        }
        
        this.apiKey = apiKey;
        this.connectButton.textContent = 'Connecting...';
        this.connectButton.disabled = true;
        
        try {
            const wsUrl = `ws://localhost:8000/ws?api_key=${encodeURIComponent(apiKey)}`;
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.saveApiKey();
                this.updateConnectionStatus(true);
                this.enableChat();
                this.connectButton.textContent = 'Disconnect';
                this.connectButton.disabled = false;
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showMessage('Connection error', 'error');
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.disableChat();
                this.connectButton.textContent = 'Connect';
                this.connectButton.disabled = false;
            };
            
        } catch (error) {
            console.error('Connection error:', error);
            this.showMessage('Failed to connect', 'error');
            this.connectButton.textContent = 'Connect';
            this.connectButton.disabled = false;
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
    
    updateConnectionStatus(connected) {
        if (connected) {
            this.connectionStatus.classList.add('connected');
            this.statusText.textContent = 'Connected';
        } else {
            this.connectionStatus.classList.remove('connected');
            this.statusText.textContent = 'Disconnected';
        }
    }
    
    enableChat() {
        this.messageInput.disabled = false;
        this.sendButton.disabled = false;
        this.voiceButton.disabled = false;
        this.messageInput.focus();
    }
    
    disableChat() {
        this.messageInput.disabled = true;
        this.sendButton.disabled = true;
        this.voiceButton.disabled = true;
    }
    
    sendMessage() {
        const text = this.messageInput.value.trim();
        if (!text || !this.isConnected) return;
        
        // Show user message
        this.showMessage(text, 'user');
        
        // Send to server
        const message = {
            type: 'text',
            data: {
                text: text
            }
        };
        
        this.ws.send(JSON.stringify(message));
        this.messageInput.value = '';
        
        // Show thinking indicator
        this.showThinkingIndicator();
    }
    
    async toggleVoiceRecording() {
        if (this.isRecording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.addEventListener('dataavailable', event => {
                this.audioChunks.push(event.data);
            });
            
            this.mediaRecorder.addEventListener('stop', async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                await this.sendAudioMessage(audioBlob);
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            });
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI
            this.voiceButton.classList.add('recording');
            this.audioVisualizer.classList.add('active');
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showMessage('Microphone access denied', 'error');
        }
    }
    
    async stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            this.voiceButton.classList.remove('recording');
            this.audioVisualizer.classList.remove('active');
        }
    }
    
    async sendAudioMessage(audioBlob) {
        // Convert blob to base64
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            
            const message = {
                type: 'audio',
                data: {
                    audio_data: base64Audio,
                    format: 'webm'
                }
            };
            
            this.ws.send(JSON.stringify(message));
            this.showThinkingIndicator();
        };
        reader.readAsDataURL(audioBlob);
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'response':
                this.handleResponse(message.data);
                break;
            case 'task_update':
                this.handleTaskUpdate(message.data);
                break;
            case 'notification':
                this.handleNotification(message.data);
                break;
            case 'status':
                this.handleStatus(message.data);
                break;
            case 'error':
                this.showMessage(message.data.error || 'An error occurred', 'error');
                break;
        }
    }
    
    handleResponse(data) {
        this.removeThinkingIndicator();
        
        // Show assistant response
        this.showMessage(data.text, 'assistant');
        
        // Play audio if available
        if (data.audio) {
            this.playAudio(data.audio);
        }
        
        // Update UI with any tool results
        if (data.tool_results) {
            console.log('Tool results:', data.tool_results);
        }
    }
    
    handleTaskUpdate(data) {
        // Update task list
        this.updateTaskList();
        
        // Show notification for important updates
        if (data.status === 'completed' && data.play_sound) {
            this.playNotificationSound();
            this.showNotification(`Task completed: ${data.message}`);
        }
    }
    
    handleNotification(data) {
        if (data.type === 'proactive_suggestion') {
            this.showMessage(data.message, 'assistant');
        } else {
            this.showNotification(data.message);
        }
    }
    
    handleStatus(data) {
        if (data.current_state) {
            // Update context display
            this.currentActivity.textContent = data.current_state.activity || '-';
            this.currentLocation.textContent = data.current_state.location || '-';
            this.currentMood.textContent = data.current_state.mood || '-';
        }
        
        if (data.active_tasks !== undefined) {
            this.renderTaskList(data.active_tasks, data.completed_tasks);
        }
    }
    
    handleQuickAction(action) {
        const actions = {
            'morning-briefing': 'Give me my morning briefing',
            'set-timer': 'Set a timer for 5 minutes',
            'take-note': 'Take a note: ',
            'check-tasks': 'What are my active tasks?'
        };
        
        const text = actions[action];
        if (text) {
            this.messageInput.value = text;
            if (!text.endsWith(': ')) {
                this.sendMessage();
            } else {
                this.messageInput.focus();
            }
        }
    }
    
    showMessage(text, type = 'system') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const messageText = document.createElement('p');
        messageText.textContent = text;
        
        messageDiv.appendChild(messageText);
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showThinkingIndicator() {
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'message assistant thinking-indicator';
        thinkingDiv.innerHTML = '<p>Thinking...</p>';
        this.messagesContainer.appendChild(thinkingDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    removeThinkingIndicator() {
        const indicator = this.messagesContainer.querySelector('.thinking-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    async updateTaskList() {
        if (!this.isConnected) return;
        
        const message = {
            type: 'command',
            data: {
                command: 'get_tasks'
            }
        };
        
        this.ws.send(JSON.stringify(message));
    }
    
    renderTaskList(activeTasks, completedTasks) {
        this.activeTasksContainer.innerHTML = '';
        
        if (!activeTasks || activeTasks.length === 0) {
            this.activeTasksContainer.innerHTML = '<p class="empty-state">No active tasks</p>';
            return;
        }
        
        activeTasks.forEach(task => {
            const taskDiv = document.createElement('div');
            taskDiv.className = `task-item ${task.status}`;
            
            const titleDiv = document.createElement('div');
            titleDiv.className = 'task-title';
            titleDiv.textContent = task.description.substring(0, 50) + '...';
            
            const statusDiv = document.createElement('div');
            statusDiv.className = 'task-status';
            statusDiv.textContent = `Status: ${task.status} - Progress: ${Math.round(task.progress * 100)}%`;
            
            taskDiv.appendChild(titleDiv);
            taskDiv.appendChild(statusDiv);
            
            this.activeTasksContainer.appendChild(taskDiv);
        });
    }
    
    showNotification(message) {
        // Check if browser supports notifications
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('Wheatley 2.0', {
                body: message,
                icon: '/favicon.ico'
            });
        }
    }
    
    async playAudio(audioData) {
        try {
            const audioBlob = new Blob([atob(audioData)], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            await audio.play();
        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }
    
    playNotificationSound() {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBi');
        audio.play();
    }
}

// Initialize the client when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.wheatley = new WheatleyClient();
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}); 