/**
 * Main JavaScript for Semantic Segmentation Web Application
 */

// Application state
const AppState = {
    training: {
        isTraining: false,
        progress: 0,
        currentEpoch: 0,
        totalEpochs: 0,
        currentLoss: 0,
        currentIoU: 0,
        bestIoU: 0,
        logs: []
    },
    dataset: {
        loaded: false,
        info: null
    },
    model: {
        loaded: false,
        name: null
    }
};

// DOM Elements
const DOM = {
    // Dataset elements
    datasetInfo: document.getElementById('dataset-info'),
    datasetStatus: document.getElementById('dataset-status'),
    
    // Training elements
    trainingForm: document.getElementById('training-form'),
    startTrainingBtn: document.getElementById('start-training-btn'),
    stopTrainingBtn: document.getElementById('stop-training-btn'),
    trainingProgress: document.getElementById('training-progress'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    currentEpoch: document.getElementById('current-epoch'),
    currentLoss: document.getElementById('current-loss'),
    currentIoU: document.getElementById('current-iou'),
    bestIoU: document.getElementById('best-iou'),
    trainingLogs: document.getElementById('training-logs'),
    
    // Prediction elements
    imageUpload: document.getElementById('image-upload'),
    predictBtn: document.getElementById('predict-btn'),
    originalImage: document.getElementById('original-image'),
    segmentedImage: document.getElementById('segmented-image'),
    blendedImage: document.getElementById('blended-image'),
    detectedClasses: document.getElementById('detected-classes'),
    predictionResults: document.getElementById('prediction-results'),
    
    // System logs
    systemLogs: document.getElementById('system-logs'),
    
    // Status indicators
    trainingStatus: document.getElementById('training-status'),
    modelStatus: document.getElementById('model-status')
};

// API Endpoints
const API = {
    DATASET_INFO: '/api/dataset/info',
    DATASET_SETUP: '/api/dataset/setup',
    TRAINING_STATUS: '/api/training/status',
    TRAINING_START: '/api/training/start',
    TRAINING_STOP: '/api/training/stop',
    MODEL_PREDICT: '/api/model/predict',
    MODEL_LIST: '/api/model/list',
    MODEL_LOAD: '/api/model/load',
    SYSTEM_HEALTH: '/api/system/health'
};

// Utility functions
const Utils = {
    // Format bytes to human readable format
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },
    
    // Format time
    formatTime(seconds) {
        if (!seconds) return '00:00:00';
        
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    },
    
    // Add log entry
    addLog(message, type = 'info', element = DOM.systemLogs) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.innerHTML = `
            <span class="log-time">[${timestamp}]</span>
            <span class="log-message">${message}</span>
        `;
        
        element.prepend(logEntry);
        
        // Keep only last 20 logs
        const logs = element.querySelectorAll('.log-entry');
        if (logs.length > 20) {
            logs[logs.length - 1].remove();
        }
    },
    
    // Update status indicator
    updateStatusIndicator(element, status, text) {
        element.className = `status-indicator status-${status}`;
        element.innerHTML = `
            <i class="fas fa-circle"></i>
            <span>${text}</span>
        `;
    },
    
    // Show notification
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type} fade-in`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
            <button class="notification-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add to notification container
        const container = document.getElementById('notifications') || (() => {
            const div = document.createElement('div');
            div.id = 'notifications';
            div.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                max-width: 400px;
            `;
            document.body.appendChild(div);
            return div;
        })();
        
        container.prepend(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    },
    
    // Format number with precision
    formatNumber(num, precision = 4) {
        if (num === null || num === undefined) return 'N/A';
        return Number(num).toFixed(precision);
    }
};

// API Service
const ApiService = {
    // Generic fetch wrapper
    async fetch(endpoint, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };
        
        const mergedOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(endpoint, mergedOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            Utils.addLog(`API Error: ${error.message}`, 'error');
            throw error;
        }
    },
    
    // Get dataset information
    async getDatasetInfo() {
        return this.fetch(API.DATASET_INFO);
    },
    
    // Setup dataset
    async setupDataset() {
        return this.fetch(API.DATASET_SETUP, {
            method: 'POST'
        });
    },
    
    // Get training status
    async getTrainingStatus() {
        return this.fetch(API.TRAINING_STATUS);
    },
    
    // Start training
    async startTraining(params) {
        return this.fetch(API.TRAINING_START, {
            method: 'POST',
            body: JSON.stringify(params)
        });
    },
    
    // Stop training
    async stopTraining() {
        return this.fetch(API.TRAINING_STOP, {
            method: 'POST'
        });
    },
    
    // Make prediction
    async predict(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        const response = await fetch(API.MODEL_PREDICT, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    },
    
    // List available models
    async listModels() {
        return this.fetch(API.MODEL_LIST);
    },
    
    // Load model
    async loadModel(modelPath) {
        return this.fetch(API.MODEL_LOAD, {
            method: 'POST',
            body: JSON.stringify({ model_path: modelPath })
        });
    },
    
    // Health check
    async healthCheck() {
        return this.fetch(API.SYSTEM_HEALTH);
    }
};

// Dataset Module
const DatasetModule = {
    async loadDatasetInfo() {
        try {
            Utils.addLog('Loading dataset information...');
            
            const data = await ApiService.getDatasetInfo();
            
            if (data.status === 'ready') {
                this.updateDatasetInfo(data);
                AppState.dataset.loaded = true;
                AppState.dataset.info = data;
                
                Utils.addLog('Dataset loaded successfully', 'success');
                Utils.updateStatusIndicator(DOM.datasetStatus, 'completed', 'Ready');
            } else {
                this.showDatasetSetupInstructions(data);
                Utils.updateStatusIndicator(DOM.datasetStatus, 'error', 'Not Found');
            }
        } catch (error) {
            Utils.addLog(`Failed to load dataset: ${error.message}`, 'error');
            Utils.updateStatusIndicator(DOM.datasetStatus, 'error', 'Error');
        }
    },
    
    updateDatasetInfo(data) {
        DOM.datasetInfo.innerHTML = `
            <div class="dataset-info">
                <div class="info-item">
                    <i class="fas fa-images"></i>
                    <div class="label">Total Images</div>
                    <div class="value">${data.total_images.toLocaleString()}</div>
                </div>
                ${data.splits ? Object.entries(data.splits).map(([split, count]) => `
                    <div class="info-item">
                        <i class="fas fa-${split === 'train' ? 'train' : split === 'val' ? 'check-circle' : 'database'}"></i>
                        <div class="label">${split.toUpperCase()}</div>
                        <div class="value">${count.toLocaleString()}</div>
                    </div>
                `).join('') : ''}
                <div class="info-item">
                    <i class="fas fa-tags"></i>
                    <div class="label">Classes</div>
                    <div class="value">${data.class_count}</div>
                </div>
            </div>
            
            <div class="classes-section">
                <h4><i class="fas fa-list"></i> Object Classes</h4>
                <div class="classes-grid">
                    ${data.classes.map((cls, idx) => `
                        <span class="class-tag" title="${cls}">
                            ${cls}
                        </span>
                    `).join('')}
                </div>
            </div>
            
            ${data.dataset_status === 'test' ? `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    Using test dataset. Download real dataset for better results.
                </div>
            ` : ''}
        `;
    },
    
    showDatasetSetupInstructions(data) {
        DOM.datasetInfo.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i>
                <h4>Dataset Not Found</h4>
                <p>${data.message || 'The Pascal VOC 2012 dataset is not available.'}</p>
                
                <div class="setup-instructions">
                    <h5>Setup Instructions:</h5>
                    <ol>
                        <li>Click the button below to download automatically</li>
                        <li>Or download manually from: 
                            <a href="https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset" target="_blank">
                                Kaggle Dataset
                            </a>
                        </li>
                        <li>Extract to: <code>data/VOCdevkit/VOC2012/</code></li>
                    </ol>
                </div>
                
                <button id="setup-dataset-btn" class="btn btn-primary">
                    <i class="fas fa-download"></i> Setup Dataset Automatically
                </button>
            </div>
        `;
        
        // Add event listener to setup button
        document.getElementById('setup-dataset-btn')?.addEventListener('click', () => {
            this.setupDataset();
        });
    },
    
    async setupDataset() {
        try {
            Utils.addLog('Setting up dataset...');
            
            const result = await ApiService.setupDataset();
            
            if (result.success) {
                Utils.showNotification('Dataset setup completed successfully!', 'success');
                Utils.addLog('Dataset setup completed', 'success');
                
                // Reload dataset info
                setTimeout(() => this.loadDatasetInfo(), 2000);
            } else {
                Utils.showNotification(`Failed to setup dataset: ${result.message}`, 'error');
                Utils.addLog(`Dataset setup failed: ${result.message}`, 'error');
            }
        } catch (error) {
            Utils.showNotification(`Error setting up dataset: ${error.message}`, 'error');
            Utils.addLog(`Dataset setup error: ${error.message}`, 'error');
        }
    }
};

// Training Module
const TrainingModule = {
    pollingInterval: null,
    
    init() {
        // Load training status
        this.loadTrainingStatus();
        
        // Start polling for updates
        this.startPolling();
        
        // Setup form submission
        DOM.trainingForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.startTraining();
        });
        
        // Setup stop button
        DOM.stopTrainingBtn?.addEventListener('click', () => {
            this.stopTraining();
        });
    },
    
    async loadTrainingStatus() {
        try {
            const data = await ApiService.getTrainingStatus();
            this.updateTrainingState(data);
        } catch (error) {
            Utils.addLog(`Failed to load training status: ${error.message}`, 'error');
        }
    },
    
    updateTrainingState(data) {
        AppState.training.isTraining = data.is_training;
        AppState.training.progress = data.progress;
        AppState.training.currentEpoch = data.current_epoch;
        AppState.training.totalEpochs = data.total_epochs;
        AppState.training.currentLoss = data.current_loss;
        AppState.training.currentIoU = data.current_iou;
        AppState.training.bestIoU = data.best_iou;
        AppState.training.logs = data.logs || [];
        
        this.updateUI();
    },
    
    updateUI() {
        // Update progress bar
        DOM.progressFill.style.width = `${AppState.training.progress}%`;
        DOM.progressText.textContent = `${Math.round(AppState.training.progress)}%`;
        
        // Update metrics
        DOM.currentEpoch.textContent = `${AppState.training.currentEpoch}/${AppState.training.totalEpochs}`;
        DOM.currentLoss.textContent = Utils.formatNumber(AppState.training.currentLoss);
        DOM.currentIoU.textContent = Utils.formatNumber(AppState.training.currentIoU, 3);
        DOM.bestIoU.textContent = Utils.formatNumber(AppState.training.bestIoU, 3);
        
        // Update training logs
        this.updateTrainingLogs();
        
        // Update buttons
        DOM.startTrainingBtn.disabled = AppState.training.isTraining;
        DOM.stopTrainingBtn.style.display = AppState.training.isTraining ? 'block' : 'none';
        
        // Update status indicator
        if (AppState.training.isTraining) {
            Utils.updateStatusIndicator(DOM.trainingStatus, 'training', 'Training');
            DOM.trainingProgress.style.display = 'block';
        } else if (AppState.training.currentEpoch > 0) {
            Utils.updateStatusIndicator(DOM.trainingStatus, 'completed', 'Completed');
            DOM.trainingProgress.style.display = 'block';
        } else {
            Utils.updateStatusIndicator(DOM.trainingStatus, 'idle', 'Idle');
            DOM.trainingProgress.style.display = 'none';
        }
    },
    
    updateTrainingLogs() {
        DOM.trainingLogs.innerHTML = AppState.training.logs
            .map(log => `<div class="log-entry info">${log}</div>`)
            .join('');
        
        // Scroll to bottom
        DOM.trainingLogs.scrollTop = DOM.trainingLogs.scrollHeight;
    },
    
    async startTraining() {
        try {
            // Get form values
            const formData = new FormData(DOM.trainingForm);
            const params = {
                epochs: parseInt(formData.get('epochs')) || 20,
                batch_size: parseInt(formData.get('batch_size')) || 4,
                learning_rate: parseFloat(formData.get('learning_rate')) || 0.001
            };
            
            Utils.addLog(`Starting training with params: ${JSON.stringify(params)}`);
            
            const result = await ApiService.startTraining(params);
            
            if (result.success) {
                Utils.showNotification('Training started successfully!', 'success');
                Utils.addLog('Training started', 'success');
            } else {
                Utils.showNotification(`Failed to start training: ${result.message}`, 'error');
                Utils.addLog(`Training start failed: ${result.message}`, 'error');
            }
        } catch (error) {
            Utils.showNotification(`Error starting training: ${error.message}`, 'error');
            Utils.addLog(`Training start error: ${error.message}`, 'error');
        }
    },
    
    async stopTraining() {
        try {
            Utils.addLog('Stopping training...');
            
            const result = await ApiService.stopTraining();
            
            if (result.success) {
                Utils.showNotification('Training stopped', 'info');
                Utils.addLog('Training stopped', 'info');
            } else {
                Utils.showNotification(`Failed to stop training: ${result.message}`, 'error');
                Utils.addLog(`Failed to stop training: ${result.message}`, 'error');
            }
        } catch (error) {
            Utils.showNotification(`Error stopping training: ${error.message}`, 'error');
            Utils.addLog(`Error stopping training: ${error.message}`, 'error');
        }
    },
    
    startPolling() {
        // Clear existing interval
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        // Poll every 2 seconds
        this.pollingInterval = setInterval(async () => {
            try {
                const data = await ApiService.getTrainingStatus();
                this.updateTrainingState(data);
            } catch (error) {
                // Silently handle polling errors
                console.debug('Polling error:', error.message);
            }
        }, 2000);
    },
    
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
};

// Prediction Module
const PredictionModule = {
    init() {
        // Setup image upload
        DOM.imageUpload?.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.previewImage(file);
            }
        });
        
        // Setup predict button
        DOM.predictBtn?.addEventListener('click', () => {
            this.predict();
        });
        
        // Setup drag and drop
        this.setupDragAndDrop();
    },
    
    setupDragAndDrop() {
        const uploadArea = DOM.imageUpload?.parentElement;
        
        if (!uploadArea) return;
        
        // Add drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        // Highlight on drag
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('drag-over');
            });
        });
        
        // Remove highlight
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('drag-over');
            });
        });
        
        // Handle drop
        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                DOM.imageUpload.files = files;
                this.previewImage(files[0]);
            }
        });
    },
    
    previewImage(file) {
        if (!file.type.match('image.*')) {
            Utils.showNotification('Please select an image file', 'error');
            return;
        }
        
        const reader = new FileReader();
        
        reader.onload = (e) => {
            DOM.originalImage.src = e.target.result;
            DOM.predictionResults.style.display = 'none';
            DOM.predictBtn.disabled = false;
        };
        
        reader.readAsDataURL(file);
    },
    
    async predict() {
        const file = DOM.imageUpload.files[0];
        
        if (!file) {
            Utils.showNotification('Please select an image first', 'error');
            return;
        }
        
        try {
            // Disable predict button
            DOM.predictBtn.disabled = true;
            DOM.predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
            Utils.addLog(`Making prediction for: ${file.name}`);
            
            const result = await ApiService.predict(file);
            
            if (result.success) {
                // Update result images
                DOM.segmentedImage.src = result.segmentation;
                DOM.blendedImage.src = result.blended;
                
                // Update detected classes
                DOM.detectedClasses.innerHTML = result.class_names
                    .map((name, idx) => `
                        <span class="class-tag">
                            ${name}
                        </span>
                    `).join('');
                
                // Show results
                DOM.predictionResults.style.display = 'block';
                
                Utils.showNotification('Prediction completed successfully!', 'success');
                Utils.addLog(`Prediction completed. Detected: ${result.class_names.join(', ')}`, 'success');
            } else {
                Utils.showNotification(`Prediction failed: ${result.message}`, 'error');
                Utils.addLog(`Prediction failed: ${result.message}`, 'error');
            }
        } catch (error) {
            Utils.showNotification(`Error during prediction: ${error.message}`, 'error');
            Utils.addLog(`Prediction error: ${error.message}`, 'error');
        } finally {
            // Re-enable predict button
            DOM.predictBtn.disabled = false;
            DOM.predictBtn.innerHTML = '<i class="fas fa-magic"></i> Predict';
        }
    }
};

// Model Module
const ModelModule = {
    async loadAvailableModels() {
        try {
            const data = await ApiService.listModels();
            
            if (data.success && data.models.length > 0) {
                AppState.model.loaded = true;
                AppState.model.name = data.models[0].name;
                
                Utils.updateStatusIndicator(DOM.modelStatus, 'completed', 'Loaded');
                Utils.addLog(`Model loaded: ${AppState.model.name}`, 'success');
            } else {
                Utils.updateStatusIndicator(DOM.modelStatus, 'idle', 'Not Loaded');
                Utils.addLog('No trained models found', 'warning');
            }
        } catch (error) {
            Utils.updateStatusIndicator(DOM.modelStatus, 'error', 'Error');
            Utils.addLog(`Failed to load models: ${error.message}`, 'error');
        }
    },
    
    async loadModel(modelPath) {
        try {
            Utils.addLog(`Loading model: ${modelPath}`);
            
            const result = await ApiService.loadModel(modelPath);
            
            if (result.success) {
                Utils.showNotification('Model loaded successfully!', 'success');
                Utils.addLog('Model loaded', 'success');
                
                // Update model state
                AppState.model.loaded = true;
                AppState.model.name = modelPath;
                
                Utils.updateStatusIndicator(DOM.modelStatus, 'completed', 'Loaded');
            } else {
                Utils.showNotification(`Failed to load model: ${result.message}`, 'error');
                Utils.addLog(`Model load failed: ${result.message}`, 'error');
            }
        } catch (error) {
            Utils.showNotification(`Error loading model: ${error.message}`, 'error');
            Utils.addLog(`Model load error: ${error.message}`, 'error');
        }
    }
};

// Application initialization
document.addEventListener('DOMContentLoaded', async () => {
    Utils.addLog('Application starting...', 'info');
    
    try {
        // Initialize modules
        await DatasetModule.loadDatasetInfo();
        TrainingModule.init();
        PredictionModule.init();
        await ModelModule.loadAvailableModels();
        
        // Check system health
        const health = await ApiService.healthCheck();
        Utils.addLog(`System health: ${health.status}`, 'success');
        
        // Show welcome message
        setTimeout(() => {
            Utils.showNotification('Semantic Segmentation Web Application Ready!', 'success');
            Utils.addLog('Application ready', 'success');
        }, 1000);
        
    } catch (error) {
        Utils.addLog(`Application initialization failed: ${error.message}`, 'error');
        Utils.showNotification('Failed to initialize application. Please check console.', 'error');
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    Utils.addLog(`Unhandled error: ${event.message}`, 'error');
    console.error('Unhandled error:', event.error);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    TrainingModule.stopPolling();
});