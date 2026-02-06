/**
 * Vision AI Studio - Enhanced Segmentation Results Focus
 */

class VisionAIStudio {
    constructor() {
        this.currentImage = null;
        this.isProcessing = false;
        this.activities = [];
        this.canvas = null;
        this.ctx = null;
        this.particles = [];
        this.animationId = null;
        this.mouseX = 0;
        this.mouseY = 0;
        this.detectedObjects = [];
        this.processingStats = {
            startTime: 0,
            endTime: 0,
            objectsCount: 0,
            confidence: 0
        };
        
        this.init();
    }

    init() {
        console.log('✨ Vision AI Studio initializing...');
        
        this.setupCanvas();
        this.setupEventListeners();
        this.startAnimations();
        this.loadInitialData();
        
        // Add initial activity
        this.addActivity('System initialized and ready');
        
        // Initialize stats
        this.updateStats();
    }

    setupCanvas() {
        this.canvas = document.getElementById('backgroundCanvas');
        if (!this.canvas) {
            console.warn('Canvas element not found');
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Setup mouse tracking
        document.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });
        
        // Initialize particles
        this.initParticles();
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    initParticles() {
        this.particles = [];
        const particleCount = Math.min(60, Math.floor((window.innerWidth * window.innerHeight) / 20000));
        
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 1.5 + 0.5,
                speedX: Math.random() * 0.2 - 0.1,
                speedY: Math.random() * 0.2 - 0.1,
                color: `rgba(79, 70, 229, ${Math.random() * 0.15 + 0.05})`
            });
        }
    }

    setupEventListeners() {
        // Analysis controls
        document.getElementById('analyzeBtn').addEventListener('click', () => this.analyzeImage());
        document.getElementById('sampleBtn').addEventListener('click', () => this.useSample());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearResults());
        document.getElementById('refreshBtn').addEventListener('click', () => this.refreshSystem());
        
        // Upload area
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('imageInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Drag and drop
        this.setupDragAndDrop(uploadArea);
        
        // Confidence threshold slider
        const confidenceSlider = document.getElementById('confidenceThreshold');
        confidenceSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            const display = e.target.parentElement.querySelector('.slider-value');
            display.textContent = `${value}%`;
            this.showSliderAnimation(e.target);
        });
        
        // Clear logs
        document.getElementById('clearLogsBtn').addEventListener('click', () => this.clearLogs());
        
        // Checkbox interactions
        this.setupCheckboxes();
        
        // Button effects
        this.setupButtonEffects();
    }

    setupDragAndDrop(uploadArea) {
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
                this.createParticleEffect(e.clientX, e.clientY, '#4f46e5', 5);
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                if (eventName === 'drop' && e.dataTransfer.files.length > 0) {
                    this.handleImageUpload({ target: { files: e.dataTransfer.files } });
                }
            });
        });
    }

    setupCheckboxes() {
        const checkboxes = document.querySelectorAll('.checkbox input');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.createParticleEffect(
                    e.target.getBoundingClientRect().left + 10,
                    e.target.getBoundingClientRect().top + 10,
                    '#4f46e5',
                    2
                );
            });
        });
    }

    setupButtonEffects() {
        document.querySelectorAll('.btn').forEach(button => {
            button.addEventListener('mouseenter', (e) => {
                this.createRipple(e, button, 'rgba(255, 255, 255, 0.3)');
                button.style.transform = 'translateY(-2px) scale(1.05)';
            });

            button.addEventListener('mouseleave', () => {
                button.style.transform = 'translateY(0) scale(1)';
            });

            button.addEventListener('click', (e) => {
                this.createRipple(e, button, 'rgba(255, 255, 255, 0.5)');
                this.createParticleEffect(e.clientX, e.clientY, '#4f46e5', 4);
            });
        });
    }

    startAnimations() {
        if (this.canvas) {
            this.animate();
        }
        
        // Floating card animations
        this.floatCards();
        
        // Pulse animations
        this.startPulseAnimations();
        
        // Subtle background animations
        this.startBackgroundAnimations();
    }

    startBackgroundAnimations() {
        // Animate glow effects
        const glows = document.querySelectorAll('.glow-effect');
        glows.forEach((glow, index) => {
            glow.style.animationDelay = `${index * 5}s`;
        });
        
        // Create occasional floating particles
        setInterval(() => {
            if (Math.random() > 0.8) {
                this.createRandomParticles('#4f46e5', 1);
            }
        }, 4000);
    }

    animate() {
        if (!this.ctx) return;
        
        // Clear canvas with subtle fade
        this.ctx.fillStyle = 'rgba(248, 250, 252, 0.02)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Update and draw particles
        this.updateParticles();
        this.drawParticles();
        
        // Draw connection lines between close particles
        this.drawConnectionLines();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    updateParticles() {
        this.particles.forEach(particle => {
            // Update position
            particle.x += particle.speedX;
            particle.y += particle.speedY;
            
            // Bounce off edges
            if (particle.x < 0 || particle.x > this.canvas.width) particle.speedX *= -1;
            if (particle.y < 0 || particle.y > this.canvas.height) particle.speedY *= -1;
            
            // Mouse interaction
            const dx = this.mouseX - particle.x;
            const dy = this.mouseY - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 100) {
                const force = 0.02;
                particle.x -= dx * force;
                particle.y -= dy * force;
            }
        });
    }

    drawParticles() {
        this.particles.forEach(particle => {
            // Draw particle with glow
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size * 2, 0, Math.PI * 2);
            this.ctx.fillStyle = particle.color.replace(')', ', 0.1)').replace('rgb', 'rgba');
            this.ctx.fill();
            
            // Draw core
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fillStyle = particle.color.replace(')', ', 0.3)').replace('rgb', 'rgba');
            this.ctx.fill();
        });
    }

    drawConnectionLines() {
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const dx = this.particles[i].x - this.particles[j].x;
                const dy = this.particles[i].y - this.particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 120) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
                    this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
                    this.ctx.strokeStyle = `rgba(79, 70, 229, ${0.08 - distance / 1500})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.stroke();
                }
            }
        }
    }

    floatCards() {
        const cards = document.querySelectorAll('.card');
        cards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
            card.classList.add('animate-fade-in');
            
            // Add hover effect
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-4px)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
            });
        });
    }

    startPulseAnimations() {
        // Status indicator pulse
        setInterval(() => {
            const pulseDot = document.querySelector('.pulse-dot');
            if (pulseDot) {
                pulseDot.style.animation = 'none';
                setTimeout(() => {
                    pulseDot.style.animation = 'pulse 1.5s infinite';
                }, 10);
            }
        }, 3000);
        
        // Upload icon pulse
        setInterval(() => {
            const uploadPulse = document.querySelector('.upload-pulse');
            if (uploadPulse) {
                uploadPulse.style.animation = 'none';
                setTimeout(() => {
                    uploadPulse.style.animation = 'pulse 2s infinite';
                }, 10);
            }
        }, 4000);
    }

    async loadInitialData() {
        try {
            await this.loadDatasetInfo();
            this.updateStats();
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }

    async loadDatasetInfo() {
        try {
            const response = await fetch('/api/dataset/info');
            const data = await response.json();
            
            const datasetInfo = document.getElementById('datasetInfo');
            datasetInfo.innerHTML = `
                <div class="dataset-stats">
                    <div class="stat-item">
                        <div class="stat-value">${data.total_images}</div>
                        <div class="stat-label">Total Images</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.class_count}</div>
                        <div class="stat-label">Classes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.training_set}</div>
                        <div class="stat-label">Training Set</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.validation_set}</div>
                        <div class="stat-label">Validation Set</div>
                    </div>
                </div>
                <div class="classes-container">
                    <h4>Object Classes:</h4>
                    <div class="classes-grid" id="classesGrid"></div>
                </div>
            `;
            
            // Add classes with animation
            const classesGrid = document.getElementById('classesGrid');
            if (Array.isArray(data.classes)) {
                data.classes.slice(0, 12).forEach((cls, index) => {
                    setTimeout(() => {
                        const div = document.createElement('div');
                        div.className = 'class-item';
                        div.textContent = typeof cls === 'object' ? cls.name : cls;
                        div.style.opacity = '0';
                        div.style.transform = 'scale(0.9)';
                        classesGrid.appendChild(div);
                        
                        setTimeout(() => {
                            div.style.transition = 'all 0.3s ease';
                            div.style.opacity = '1';
                            div.style.transform = 'scale(1)';
                        }, 10);
                    }, index * 30);
                });
            }
            
            this.addActivity('Dataset information loaded');
            
        } catch (error) {
            console.error('Failed to load dataset info:', error);
            this.addActivity('Failed to load dataset info');
        }
    }

    handleImageUpload(event) {
        const file = event.target.files?.[0] || event.dataTransfer?.files?.[0];
        if (!file) return;
        
        if (!file.type.match('image.*')) {
            this.showNotification('Please select an image file (JPG, PNG, or WebP)', 'error');
            return;
        }
        
        if (file.size > 5 * 1024 * 1024) {
            this.showNotification('Image size must be less than 5MB', 'error');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById('originalImage');
            const placeholder = document.getElementById('originalPlaceholder');
            
            // Hide placeholder
            placeholder.style.display = 'none';
            
            // Fade in animation
            img.style.opacity = '0';
            img.style.transform = 'scale(0.95)';
            
            setTimeout(() => {
                img.src = e.target.result;
                img.style.display = 'block';
                
                // Animate image
                setTimeout(() => {
                    img.style.transition = 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
                    img.style.opacity = '1';
                    img.style.transform = 'scale(1)';
                }, 100);
                
                // Get image dimensions
                const tempImg = new Image();
                tempImg.onload = () => {
                    document.getElementById('originalSize').textContent = `${tempImg.width}×${tempImg.height}`;
                    document.getElementById('originalWeight').textContent = `${Math.round(file.size / 1024)}KB`;
                };
                tempImg.src = e.target.result;
            }, 300);
            
            this.currentImage = file;
            document.getElementById('analyzeBtn').disabled = false;
            
            this.showNotification('Image uploaded successfully!', 'success');
            this.addActivity(`Uploaded image: ${file.name}`);
            
            // Create upload success particle effect
            const uploadZone = document.getElementById('uploadArea');
            const rect = uploadZone.getBoundingClientRect();
            this.createParticleEffect(
                rect.left + rect.width / 2,
                rect.top + rect.height / 2,
                '#4f46e5',
                8
            );
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        if (!this.currentImage) {
            this.showNotification('Please upload an image first!', 'warning');
            return;
        }
        
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        this.processingStats.startTime = Date.now();
        
        const analyzeBtn = document.getElementById('analyzeBtn');
        const originalContent = analyzeBtn.innerHTML;
        
        // Show loading state
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
        analyzeBtn.disabled = true;
        
        // Show scanning animation
        this.showScanningAnimation();
        
        // Hide result placeholder
        document.getElementById('resultPlaceholder').style.display = 'none';
        
        // Simulate processing with mock data
        setTimeout(async () => {
            try {
                // In a real app, this would be an API call
                const mockResult = await this.generateMockResults();
                
                // Update result image
                const resultImg = document.getElementById('resultImage');
                resultImg.style.opacity = '0';
                resultImg.style.transform = 'scale(0.95)';
                
                setTimeout(() => {
                    resultImg.src = mockResult.result;
                    resultImg.style.display = 'block';
                    
                    // Animate result image
                    setTimeout(() => {
                        resultImg.style.transition = 'all 0.7s cubic-bezier(0.34, 1.56, 0.64, 1)';
                        resultImg.style.opacity = '1';
                        resultImg.style.transform = 'scale(1)';
                    }, 100);
                }, 300);
                
                // Update processing stats
                this.processingStats.endTime = Date.now();
                this.processingStats.objectsCount = mockResult.detected.length;
                this.processingStats.confidence = mockResult.confidence;
                
                // Update detected objects
                this.detectedObjects = mockResult.detected;
                this.updateDetectedObjects();
                
                // Update statistics
                this.updateStats();
                this.updateChart();
                
                this.showNotification('Analysis completed successfully!', 'success');
                this.addActivity(`Analysis completed: ${mockResult.detected.length} objects detected`);
                
                // Create celebration particle effect
                const resultContainer = document.querySelector('.image-container-large:last-child');
                const rect = resultContainer.getBoundingClientRect();
                this.createParticleEffect(
                    rect.left + rect.width / 2,
                    rect.top + rect.height / 2,
                    '#059669',
                    15
                );
                
            } catch (error) {
                this.showNotification('Analysis failed: ' + error.message, 'error');
                this.addActivity(`Analysis failed: ${error.message}`);
            } finally {
                analyzeBtn.innerHTML = originalContent;
                analyzeBtn.disabled = false;
                this.isProcessing = false;
            }
        }, 1500);
    }

    async generateMockResults() {
        // Generate mock segmentation results
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Load original image to get dimensions
        const originalImg = document.getElementById('originalImage');
        canvas.width = 400;
        canvas.height = 300;
        
        // Create mock segmentation
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add segmentation overlays
        const colors = ['#4f46e5', '#7c3aed', '#059669', '#dc2626', '#d97706'];
        const classes = ['Person', 'Car', 'Dog', 'Chair', 'Table', 'Bottle', 'Bird', 'Bicycle'];
        
        const detected = [];
        for (let i = 0; i < 5; i++) {
            const x = Math.random() * canvas.width;
            const y = Math.random() * canvas.height;
            const width = Math.random() * 100 + 50;
            const height = Math.random() * 80 + 40;
            const color = colors[i];
            const className = classes[Math.floor(Math.random() * classes.length)];
            const confidence = Math.random() * 0.3 + 0.7;
            
            // Draw segmentation area
            ctx.fillStyle = color + '40';
            ctx.fillRect(x, y, width, height);
            
            // Draw border
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);
            
            // Draw label
            ctx.fillStyle = color;
            ctx.font = 'bold 12px Inter';
            ctx.fillText(`${className} ${Math.round(confidence * 100)}%`, x + 5, y - 5);
            
            detected.push({
                name: className,
                confidence: confidence,
                color: color,
                area: width * height
            });
        }
        
        // Blend with original image
        ctx.globalAlpha = 0.3;
        ctx.drawImage(originalImg, 0, 0, canvas.width, canvas.height);
        
        return {
            result: canvas.toDataURL('image/jpeg', 0.9),
            detected: detected,
            confidence: Math.random() * 0.2 + 0.8,
            processingTime: Math.random() * 500 + 300
        };
    }

    updateDetectedObjects() {
        const container = document.getElementById('detectedItems');
        container.innerHTML = '';
        
        if (this.detectedObjects.length > 0) {
            this.detectedObjects.forEach((obj, index) => {
                setTimeout(() => {
                    const div = document.createElement('div');
                    div.className = 'object-item';
                    div.style.backgroundColor = obj.color;
                    div.innerHTML = `
                        <div class="object-name">${obj.name}</div>
                        <div class="object-confidence">${Math.round(obj.confidence * 100)}%</div>
                    `;
                    div.style.opacity = '0';
                    div.style.transform = 'translateY(10px) scale(0.9)';
                    container.appendChild(div);
                    
                    setTimeout(() => {
                        div.style.transition = 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)';
                        div.style.opacity = '1';
                        div.style.transform = 'translateY(0) scale(1)';
                    }, 10);
                    
                    // Add click effect
                    div.addEventListener('click', () => {
                        this.highlightObject(obj);
                    });
                }, index * 100);
            });
        } else {
            const div = document.createElement('div');
            div.className = 'object-item';
            div.style.backgroundColor = '#94a3b8';
            div.innerHTML = `
                <div class="object-name">No objects</div>
                <div class="object-confidence">--%</div>
            `;
            container.appendChild(div);
        }
    }

    highlightObject(obj) {
        // Create particle effect on object click
        const objects = document.querySelectorAll('.object-item');
        objects.forEach(item => {
            if (item.querySelector('.object-name').textContent === obj.name) {
                const rect = item.getBoundingClientRect();
                this.createParticleEffect(
                    rect.left + rect.width / 2,
                    rect.top + rect.height / 2,
                    obj.color,
                    8
                );
                
                // Add bounce animation
                item.style.animation = 'none';
                setTimeout(() => {
                    item.style.animation = 'bounce 0.5s ease';
                }, 10);
            }
        });
        
        this.showNotification(`Selected: ${obj.name} (${Math.round(obj.confidence * 100)}% confidence)`, 'info');
    }

    updateStats() {
        const processingTime = this.processingStats.endTime - this.processingStats.startTime;
        
        // Update main stats
        document.getElementById('objectsCount').textContent = this.processingStats.objectsCount;
        document.getElementById('processingTime').textContent = processingTime > 0 ? `${Math.round(processingTime)}ms` : '0s';
        document.getElementById('modelConfidence').textContent = this.processingStats.confidence > 0 ? 
            `${Math.round(this.processingStats.confidence * 100)}%` : '0%';
        
        // Update detailed stats
        document.getElementById('processingSpeed').textContent = processingTime > 0 ? `${Math.round(processingTime)}ms` : '0ms';
        document.getElementById('confidenceScore').textContent = this.processingStats.confidence > 0 ? 
            `${Math.round(this.processingStats.confidence * 100)}%` : '0%';
        
        // Update detection summary
        document.getElementById('totalClasses').textContent = new Set(this.detectedObjects.map(obj => obj.name)).size;
        document.getElementById('detectionRate').textContent = this.detectedObjects.length > 0 ? 
            `${Math.round((this.detectedObjects.length / 8) * 100)}%` : '0%';
        
        const avgConfidence = this.detectedObjects.length > 0 ?
            this.detectedObjects.reduce((sum, obj) => sum + obj.confidence, 0) / this.detectedObjects.length : 0;
        document.getElementById('avgConfidence').textContent = `${Math.round(avgConfidence * 100)}%`;
        
        // Update detailed statistics
        this.updateDetailedStats(avgConfidence);
    }

    updateDetailedStats(avgConfidence) {
        // Mock detailed statistics
        const precision = avgConfidence * 0.9 + Math.random() * 0.1;
        const recall = avgConfidence * 0.85 + Math.random() * 0.15;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        const iou = avgConfidence * 0.8 + Math.random() * 0.2;
        
        // Update values
        document.getElementById('precisionScore').textContent = precision.toFixed(2);
        document.getElementById('recallRate').textContent = `${Math.round(recall * 100)}%`;
        document.getElementById('f1Score').textContent = f1.toFixed(2);
        document.getElementById('iouScore').textContent = `${Math.round(iou * 100)}%`;
        
        // Update progress bars
        document.getElementById('precisionFill').style.width = `${precision * 100}%`;
        document.getElementById('recallFill').style.width = `${recall * 100}%`;
        document.getElementById('f1Fill').style.width = `${f1 * 100}%`;
        document.getElementById('iouFill').style.width = `${iou * 100}%`;
    }

    updateChart() {
        const chartBars = document.querySelector('.chart-bars');
        const chartLegend = document.querySelector('.chart-legend');
        
        if (!chartBars || !chartLegend) return;
        
        chartBars.innerHTML = '';
        chartLegend.innerHTML = '';
        
        // Group objects by class
        const classCounts = {};
        this.detectedObjects.forEach(obj => {
            classCounts[obj.name] = (classCounts[obj.name] || 0) + 1;
        });
        
        const classes = Object.keys(classCounts);
        const maxCount = Math.max(...Object.values(classCounts));
        
        // Create chart bars
        classes.forEach((className, index) => {
            const count = classCounts[className];
            const percentage = (count / maxCount) * 100;
            const color = this.detectedObjects.find(obj => obj.name === className)?.color || '#4f46e5';
            
            // Create bar
            const bar = document.createElement('div');
            bar.className = 'chart-bar';
            bar.style.height = `${percentage}%`;
            bar.style.backgroundColor = color;
            bar.setAttribute('data-value', count);
            chartBars.appendChild(bar);
            
            // Create legend item
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
                <div class="legend-color" style="background-color: ${color}"></div>
                <span>${className} (${count})</span>
            `;
            chartLegend.appendChild(legendItem);
        });
    }

    useSample() {
        const sampleImages = [
            'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=800&h=600&fit=crop',
            'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=800&h=600&fit=crop',
            'https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=800&h=600&fit=crop'
        ];
        
        const randomImage = sampleImages[Math.floor(Math.random() * sampleImages.length)];
        const img = document.getElementById('originalImage');
        const placeholder = document.getElementById('originalPlaceholder');
        
        // Hide placeholder
        placeholder.style.display = 'none';
        
        // Fade out animation
        img.style.transition = 'opacity 0.3s ease';
        img.style.opacity = '0';
        
        setTimeout(() => {
            img.src = randomImage;
            img.style.display = 'block';
            
            // Fade in with bounce animation
            setTimeout(() => {
                img.style.transition = 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
                img.style.opacity = '1';
                img.style.transform = 'scale(1.05)';
                
                setTimeout(() => {
                    img.style.transform = 'scale(1)';
                }, 500);
            }, 100);
            
            // Create mock file object
            this.currentImage = new File([], 'sample.jpg', { type: 'image/jpeg' });
            document.getElementById('analyzeBtn').disabled = false;
            
            // Reset result image
            document.getElementById('resultImage').style.display = 'none';
            document.getElementById('resultPlaceholder').style.display = 'flex';
            
            // Reset stats
            this.resetResults();
            
            this.showNotification('Sample image loaded!', 'info');
            this.addActivity('Loaded sample image');
            
            // Create sample load particle effect
            const sampleBtn = document.getElementById('sampleBtn');
            const rect = sampleBtn.getBoundingClientRect();
            this.createParticleEffect(
                rect.left + rect.width / 2,
                rect.top + rect.height / 2,
                '#7c3aed',
                6
            );
            
            // Update image info
            const tempImg = new Image();
            tempImg.onload = () => {
                document.getElementById('originalSize').textContent = `${tempImg.width}×${tempImg.height}`;
                document.getElementById('originalWeight').textContent = '256KB'; // Mock size
            };
            tempImg.src = randomImage;
            
        }, 300);
    }

    clearResults() {
        // Reset original image
        const originalImg = document.getElementById('originalImage');
        const originalPlaceholder = document.getElementById('originalPlaceholder');
        
        originalImg.style.opacity = '0';
        originalImg.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            originalImg.src = '';
            originalImg.style.display = 'none';
            originalPlaceholder.style.display = 'flex';
            
            // Reset result image
            const resultImg = document.getElementById('resultImage');
            const resultPlaceholder = document.getElementById('resultPlaceholder');
            
            resultImg.style.opacity = '0';
            resultImg.style.transform = 'scale(0.95)';
            
            setTimeout(() => {
                resultImg.src = '';
                resultImg.style.display = 'none';
                resultPlaceholder.style.display = 'flex';
            }, 150);
            
            // Reset stats
            this.resetResults();
            
            // Disable analyze button
            document.getElementById('analyzeBtn').disabled = true;
            
            this.currentImage = null;
            
            this.showNotification('Results cleared', 'info');
            this.addActivity('Cleared all results');
            
            // Create clear particle effect
            const clearBtn = document.getElementById('clearBtn');
            const rect = clearBtn.getBoundingClientRect();
            this.createParticleEffect(
                rect.left + rect.width / 2,
                rect.top + rect.height / 2,
                '#dc2626',
                8
            );
            
        }, 300);
    }

    resetResults() {
        // Reset all statistics
        this.detectedObjects = [];
        this.processingStats = {
            startTime: 0,
            endTime: 0,
            objectsCount: 0,
            confidence: 0
        };
        
        // Update UI
        this.updateStats();
        this.updateDetectedObjects();
        this.updateChart();
        
        // Clear containers
        document.getElementById('detectedItems').innerHTML = '';
        document.querySelector('.chart-bars').innerHTML = '';
        document.querySelector('.chart-legend').innerHTML = '';
        
        // Reset progress bars
        ['precisionFill', 'recallFill', 'f1Fill', 'iouFill'].forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.width = '0%';
        });
    }

    refreshSystem() {
        this.showNotification('Refreshing system...', 'info');
        this.addActivity('System refresh initiated');
        
        // Create refresh animation
        const refreshBtn = document.getElementById('refreshBtn');
        refreshBtn.style.animation = 'spin 1s linear';
        
        setTimeout(() => {
            refreshBtn.style.animation = '';
            this.showNotification('System refreshed successfully!', 'success');
            this.addActivity('System refresh completed');
            
            // Create particle effect
            const rect = refreshBtn.getBoundingClientRect();
            this.createParticleEffect(
                rect.left + rect.width / 2,
                rect.top + rect.height / 2,
                '#4f46e5',
                12
            );
        }, 1000);
    }

    // Animation Helper Methods
    createRipple(event, element, color) {
        const ripple = document.createElement('div');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: ${color};
            transform: scale(0);
            animation: ripple 0.6s linear;
            width: ${size}px;
            height: ${size}px;
            top: ${y}px;
            left: ${x}px;
            pointer-events: none;
            z-index: 1;
        `;
        
        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    }

    createParticleEffect(x, y, color, count = 6) {
        const particlesContainer = document.getElementById('particlesContainer');
        if (!particlesContainer) return;
        
        for (let i = 0; i < count; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: fixed;
                width: 4px;
                height: 4px;
                background: ${color};
                border-radius: 50%;
                pointer-events: none;
                z-index: 9998;
                top: ${y}px;
                left: ${x}px;
                animation: particleFloat 1s ease-out forwards;
                box-shadow: 0 0 10px ${color};
            `;
            
            // Set random trajectory
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * 50 + 30;
            particle.style.setProperty('--tx', `${Math.cos(angle) * distance}px`);
            particle.style.setProperty('--ty', `${Math.sin(angle) * distance}px`);
            
            particlesContainer.appendChild(particle);
            
            setTimeout(() => {
                if (particle.parentNode) {
                    particle.remove();
                }
            }, 1000);
        }
    }

    createRandomParticles(color, count = 3) {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        for (let i = 0; i < count; i++) {
            const x = Math.random() * width;
            const y = Math.random() * height;
            this.createParticleEffect(x, y, color, 1);
        }
    }

    showSliderAnimation(slider) {
        const rect = slider.getBoundingClientRect();
        const value = (slider.value / slider.max) * 100;
        
        // Update slider track
        const track = slider.parentElement.querySelector('.slider-track');
        if (track) {
            track.style.setProperty('--slider-value', `${value}%`);
        }
        
        // Create particle effect at slider thumb
        const thumbX = rect.left + (rect.width * (slider.value / slider.max));
        const thumbY = rect.top + rect.height / 2;
        this.createParticleEffect(thumbX, thumbY, '#4f46e5', 2);
    }

    showScanningAnimation() {
        const scanner = document.createElement('div');
        scanner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                #4f46e5 50%, 
                transparent 100%);
            z-index: 9999;
            animation: scan 2s ease-out;
            pointer-events: none;
        `;
        
        document.body.appendChild(scanner);
        setTimeout(() => scanner.remove(), 2000);
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notificationContainer');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const icons = {
            success: '<i class="fas fa-check-circle"></i>',
            error: '<i class="fas fa-exclamation-circle"></i>',
            warning: '<i class="fas fa-exclamation-triangle"></i>',
            info: '<i class="fas fa-info-circle"></i>'
        };
        
        notification.innerHTML = `
            ${icons[type] || icons.info}
            <span>${message}</span>
            <button class="notification-close"><i class="fas fa-times"></i></button>
        `;
        
        container.appendChild(notification);
        
        // Add notification styles if not present
        if (!document.querySelector('#notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                .notification {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(226, 232, 240, 0.8);
                    border-radius: 12px;
                    padding: 1rem 1.25rem;
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    animation: slideInRight 0.3s ease, fadeOut 0.3s ease 2.7s forwards;
                    max-width: 400px;
                    box-shadow: 0 10px 25px rgba(100, 116, 139, 0.15);
                }
                
                .notification-success {
                    border-left: 4px solid #059669;
                }
                
                .notification-error {
                    border-left: 4px solid #dc2626;
                }
                
                .notification-warning {
                    border-left: 4px solid #d97706;
                }
                
                .notification-info {
                    border-left: 4px solid #4f46e5;
                }
                
                .notification i:first-child {
                    font-size: 1.2rem;
                }
                
                .notification-success i:first-child {
                    color: #059669;
                }
                
                .notification-error i:first-child {
                    color: #dc2626;
                }
                
                .notification-warning i:first-child {
                    color: #d97706;
                }
                
                .notification-info i:first-child {
                    color: #4f46e5;
                }
                
                .notification span {
                    flex: 1;
                    color: var(--darker);
                    font-size: 0.95rem;
                }
                
                .notification-close {
                    background: transparent;
                    border: none;
                    color: var(--gray);
                    cursor: pointer;
                    padding: 0.25rem;
                    border-radius: 4px;
                    transition: var(--transition);
                }
                
                .notification-close:hover {
                    background: var(--light-gray);
                    color: var(--darker);
                }
            `;
            document.head.appendChild(style);
        }
        
        // Close button
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            notification.style.animation = 'fadeOut 0.3s ease forwards';
            setTimeout(() => notification.remove(), 300);
        });
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'fadeOut 0.3s ease forwards';
                setTimeout(() => notification.remove(), 300);
            }
        }, 3000);
    }

    addActivity(message) {
        const activityLog = document.getElementById('activityLog');
        if (!activityLog) return;
        
        const now = new Date();
        const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <div class="log-time">${timeString}</div>
            <div class="log-content">${message}</div>
        `;
        
        // Add animation
        entry.style.animation = 'fadeIn 0.3s ease';
        
        activityLog.insertBefore(entry, activityLog.firstChild);
        
        // Keep only last 15 entries
        const entries = activityLog.querySelectorAll('.log-entry');
        if (entries.length > 15) {
            activityLog.removeChild(entries[entries.length - 1]);
        }
        
        // Store in state
        this.activities.unshift({
            message,
            time: now
        });
        
        // Auto scroll to top
        activityLog.scrollTop = 0;
        
        // Add subtle particle effect
        this.createParticleEffect(
            activityLog.getBoundingClientRect().left + 20,
            activityLog.getBoundingClientRect().top + 10,
            '#4f46e5',
            1
        );
    }

    clearLogs() {
        const activityLog = document.getElementById('activityLog');
        if (!activityLog) return;
        
        // Fade out animation
        activityLog.style.transition = 'opacity 0.3s ease';
        activityLog.style.opacity = '0';
        
        setTimeout(() => {
            activityLog.innerHTML = `
                <div class="log-entry">
                    <div class="log-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                    <div class="log-content">Log cleared</div>
                </div>
            `;
            activityLog.style.opacity = '1';
            activityLog.style.transition = 'opacity 0.3s ease';
            
            this.activities = [];
            this.showNotification('Activity log cleared', 'info');
        }, 300);
    }

    cleanup() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.visionAI = new VisionAIStudio();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (window.visionAI) {
            window.visionAI.cleanup();
        }
    });
});

// Add missing animations
const animationStyles = document.createElement('style');
animationStyles.textContent = `
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .dragover {
        animation: pulse 0.5s ease-in-out;
    }
    
    .fa-spinner {
        animation: spin 1s linear infinite;
    }
`;
document.head.appendChild(animationStyles);

console.log('🎨 Vision AI Studio loaded successfully!');