// ============================================================================
// WEB SPEECH API - Real-time Voice Typing (Google Voice Typing Style)
// ============================================================================

let recognition = null;
let isListening = false;

function initSpeechRecognition() {
    // Check browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        console.warn('Speech Recognition not supported in this browser');
        return false;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = true; // Keep listening
    recognition.interimResults = true; // Show results as you speak
    recognition.lang = 'en-US'; // Default language
    
    // Handle results
    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            
            if (event.results[i].isFinal) {
                // Final result - add to transcription
                finalTranscript += transcript;
            } else {
                // Intermediate result - show as typing
                interimTranscript += transcript;
            }
        }
        
        // Update live display
        if (interimTranscript) {
            showLiveTyping(interimTranscript);
        }
        
        if (finalTranscript) {
            addToTranscription(finalTranscript);
        }
    };
    
    // Handle errors
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'no-speech') {
            // No speech detected, continue listening
        }
    };
    
    // Handle end - auto restart for continuous listening
    recognition.onend = () => {
        if (isListening) {
            // Auto restart if still recording
            recognition.start();
        }
    };
    
    return true;
}

function startSpeechRecognition(language = 'en-US') {
    if (!recognition) {
        if (!initSpeechRecognition()) {
            return false;
        }
    }
    
    // Set language
    recognition.lang = language;
    
    try {
        recognition.start();
        isListening = true;
        console.log('Speech recognition started');
        return true;
    } catch (e) {
        console.error('Failed to start speech recognition:', e);
        return false;
    }
}

function stopSpeechRecognition() {
    if (recognition && isListening) {
        isListening = false;
        recognition.stop();
        console.log('Speech recognition stopped');
    }
}

function showLiveTyping(text) {
    let liveEl = document.getElementById('live-typing');
    if (!liveEl) {
        liveEl = document.createElement('div');
        liveEl.id = 'live-typing';
        liveEl.style.cssText = 'padding: 12px; background: rgba(37, 99, 235, 0.1); border-radius: 8px; margin-bottom: 12px; border-left: 3px solid #2563EB; animation: pulse 1s infinite;';
        transcriptionBody.insertBefore(liveEl, transcriptionBody.firstChild);
    }
    
    liveEl.innerHTML = `
        <div style="font-size: 11px; color: #2563EB; margin-bottom: 4px; font-weight: 600;">
            <i class="fa-solid fa-circle" style="animation: pulse 1s infinite; margin-right: 4px;"></i>
            Listening...
        </div>
        <div style="font-size: 14px; color: var(--text-primary); font-style: italic;">${text}...</div>
    `;
}

function addToTranscription(text) {
    // Remove live typing indicator
    const liveEl = document.getElementById('live-typing');
    if (liveEl) {
        liveEl.remove();
    }
    
    // Add to segments
    const startTime = appState.secondsElapsed;
    const endTime = startTime + 1; // Approximate
    
    appState.transcribedSegments.push({
        timestamp: `00:${String(startTime).padStart(2, '0')} - 00:${String(endTime).padStart(2, '0')}`,
        text: text.trim(),
        start: startTime,
        end: endTime
    });
    
    // Update UI
    renderTranscription(appState.transcribedSegments);
    
    // Auto-scroll
    transcriptionBody.scrollTop = transcriptionBody.scrollHeight;
}

// ============================================================================
// APP STATE
// ============================================================================

const appState = {
    isRecording: false,
    isPaused: false,
    recorder: null,
    timerInterval: null,
    secondsElapsed: 0,
    currentSessionId: null,
    transcribedSegments: [],
    analyserNode: null,
    animationFrameId: null,
    detectedLanguage: null
};

let allSessionsList = [];

// UI Elements
const recordBtn = document.getElementById('record-btn');
const pauseBtn = document.getElementById('pause-btn');
const stopBtn = document.getElementById('stop-btn');
const timerEl = document.getElementById('recording-timer');
const statusLabel = document.getElementById('status-label');
const statusDot = document.getElementById('status-indicator-dot');
const languageSelect = document.getElementById('language-select');
const transcriptionBody = document.getElementById('transcription-body');
const summaryBody = document.getElementById('summary-body');
const autosaveIndicator = document.getElementById('autosave-indicator');
const sessionsList = document.getElementById('sessions-list');
const refreshSessionsBtn = document.getElementById('refresh-sessions-btn');
const copyTextBtn = document.getElementById('copy-text-btn');
const downloadDocxBtn = document.getElementById('download-docx-btn');
const downloadPdfBtn = document.getElementById('download-pdf-btn');
const clearTextBtn = document.getElementById('clear-text-btn');
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.getElementById('sidebar');
const waveformCanvas = document.getElementById('waveform-canvas');
const canvasCtx = waveformCanvas.getContext('2d');
const historySearchInput = document.getElementById('history-search');
const transcriptionSearch = document.getElementById('transcription-search');

// ============================================================================
// AUDIO VISUALIZER
// ============================================================================

function resizeCanvas() {
    waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
    waveformCanvas.height = waveformCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawFlatLine() {
    canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    canvasCtx.fillStyle = '#FFFFFF';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    canvasCtx.lineWidth = 2.5;
    canvasCtx.strokeStyle = 'rgba(37, 99, 235, 0.35)';
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, waveformCanvas.height / 2);
    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();
}
drawFlatLine();

function visualize() {
    if (!appState.isRecording) return;
    
    appState.animationFrameId = requestAnimationFrame(visualize);
    
    if (!appState.analyserNode) return;
    
    const bufferLength = appState.analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    if (appState.isPaused) {
        drawFlatLine();
        return;
    }
    
    appState.analyserNode.getByteTimeDomainData(dataArray);
    
    canvasCtx.fillStyle = '#FFFFFF';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    
    canvasCtx.lineWidth = 2.5;
    
    const gradient = canvasCtx.createLinearGradient(0, 0, waveformCanvas.width, 0);
    gradient.addColorStop(0, '#2563EB');
    gradient.addColorStop(0.5, '#8B5CF6');
    gradient.addColorStop(1, '#2563EB');
    
    canvasCtx.strokeStyle = gradient;
    canvasCtx.beginPath();
    
    const sliceWidth = waveformCanvas.width * 1.0 / bufferLength;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * waveformCanvas.height / 2;
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        x += sliceWidth;
    }
    
    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();
}

// ============================================================================
// TIMER
// ============================================================================

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function startTimer() {
    appState.secondsElapsed = 0;
    timerEl.textContent = "00:00";
    appState.timerInterval = setInterval(() => {
        if (!appState.isPaused) {
            appState.secondsElapsed++;
            timerEl.textContent = formatTime(appState.secondsElapsed);
        }
    }, 1000);
}

function stopTimer() {
    clearInterval(appState.timerInterval);
    appState.timerInterval = null;
}

// ============================================================================
// RECORDING CONTROLS
// ============================================================================

async function startRecording() {
    try {
        resizeCanvas();
        
        // Start audio recorder for visualization
        appState.recorder = new WavRecorder();
        await appState.recorder.start((analyser) => {
            appState.analyserNode = analyser;
        });

        // Set session ID
        const now = new Date();
        const dateStr = now.toISOString().slice(0, 10).replace(/-/g, '');
        const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '');
        appState.currentSessionId = `session_${dateStr}_${timeStr}`;

        // Reset state
        appState.isRecording = true;
        appState.isPaused = false;
        appState.transcribedSegments = [];
        
        // Start Web Speech API for real-time transcription
        const langMap = {
            'auto': 'en-US',
            'en': 'en-US',
            'hi': 'hi-IN',
            'ur': 'ur-PK',
            'bn': 'bn-BD',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'zh': 'zh-CN',
            'ar': 'ar-SA',
            'pt': 'pt-BR',
            'ru': 'ru-RU',
            'ja': 'ja-JP',
            'ko': 'ko-KR'
        };
        
        const speechLang = langMap[languageSelect.value] || 'en-US';
        
        if (!startSpeechRecognition(speechLang)) {
            console.warn('Web Speech API not available, using fallback');
        }

        // UI updates
        recordBtn.classList.add('recording');
        recordBtn.title = "Recording... Click Stop to finish";
        pauseBtn.disabled = false;
        stopBtn.disabled = false;
        languageSelect.disabled = true;
        
        statusLabel.textContent = "Live transcription active";
        statusLabel.style.color = "var(--color-green)";
        statusDot.className = "status-indicator-dot active";
        statusDot.style.backgroundColor = "var(--color-green)";
        
        transcriptionBody.innerHTML = `
            <div class="loading-placeholder">
                <i class="fa-solid fa-circle-notch fa-spin text-blue-600"></i>
                <p>Speak now - text will appear instantly...</p>
            </div>
        `;
        
        startTimer();
        visualize();
        
        autosaveIndicator.style.visibility = "visible";
        autosaveIndicator.innerHTML = '<i class="fa-solid fa-circle-check text-green-600"></i> <span>Live transcription active</span>';
        
    } catch (err) {
        console.error("Failed to start recording:", err);
        statusLabel.textContent = `Error: ${err.message}`;
        statusLabel.style.color = "var(--color-red)";
    }
}

function togglePause() {
    if (!appState.isRecording) return;
    
    if (appState.isPaused) {
        appState.recorder.resume();
        appState.isPaused = false;
        pauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>Pause</span>';
        
        // Resume speech recognition
        startSpeechRecognition(languageSelect.value);
        
        statusLabel.textContent = "Live transcription active";
        statusLabel.style.color = "var(--color-green)";
        statusDot.className = "status-indicator-dot active";
        statusDot.style.backgroundColor = "var(--color-green)";
    } else {
        appState.recorder.pause();
        appState.isPaused = true;
        pauseBtn.innerHTML = '<i class="fa-solid fa-play"></i> <span>Resume</span>';
        
        // Pause speech recognition
        stopSpeechRecognition();
        
        statusLabel.textContent = "Paused";
        statusLabel.style.color = "var(--color-amber)";
        statusDot.style.backgroundColor = "var(--color-amber)";
    }
}

async function stopRecording() {
    if (!appState.isRecording) return;
    
    // Stop speech recognition
    stopSpeechRecognition();
    
    // Stop timers and visuals
    stopTimer();
    cancelAnimationFrame(appState.animationFrameId);
    
    // Stop recorder
    appState.recorder.stop();
    appState.isRecording = false;
    appState.isPaused = false;
    
    // Reset UI
    recordBtn.classList.remove('recording');
    recordBtn.title = "Start Recording";
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    languageSelect.disabled = false;
    
    drawFlatLine();
    
    // Save if we have transcription
    if (appState.transcribedSegments.length > 0) {
        await saveSession(true);
        statusLabel.textContent = "Session saved!";
        statusLabel.style.color = "var(--color-green)";
    } else {
        statusLabel.textContent = "Ready to record";
        statusLabel.style.color = "var(--text-secondary)";
        statusDot.style.backgroundColor = "var(--text-muted)";
    }
}

// ============================================================================
// SAVE & LOAD
// ============================================================================

async function saveSession(isFinal = false) {
    try {
        const currentUser = JSON.parse(localStorage.getItem('auraScribeCurrentUser'));
        const userId = currentUser ? currentUser.id : 'guest';
        
        const response = await fetch("/api/save", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                session_id: appState.currentSessionId,
                text_segments: appState.transcribedSegments,
                language: languageSelect.value,
                user_id: userId
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            autosaveIndicator.innerHTML = '<i class="fa-solid fa-circle-check text-green-600"></i> <span>Saved!</span>';
            downloadDocxBtn.disabled = false;
            
            if (data.summary_points && data.summary_points.length > 0) {
                renderSummary(data.summary_points);
                downloadPdfBtn.disabled = false;
            }
            
            if (isFinal) {
                loadSessionHistory();
            }
        }
    } catch (err) {
        console.error("Save error:", err);
    }
}

function renderTranscription(segments) {
    if (!segments || segments.length === 0) {
        transcriptionBody.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon"><i class="fa-solid fa-microphone-slash"></i></div>
                <h3>No transcription yet</h3>
                <p>Speak and text will appear instantly.</p>
            </div>
        `;
        return;
    }
    
    const query = transcriptionSearch.value.trim().toLowerCase();
    transcriptionBody.innerHTML = "";
    
    segments.forEach((seg, idx) => {
        const segDiv = document.createElement("div");
        segDiv.className = "transcription-segment";
        
        const timeDiv = document.createElement("div");
        timeDiv.className = "segment-time";
        timeDiv.textContent = seg.timestamp || "00:00 - 00:00";
        
        const textDiv = document.createElement("div");
        textDiv.className = "segment-text";
        textDiv.contentEditable = "true";
        textDiv.textContent = seg.text;
        
        if (query !== "" && !seg.text.toLowerCase().includes(query)) {
            segDiv.style.opacity = "0.35";
        }
        
        textDiv.onblur = () => {
            appState.transcribedSegments[idx].text = textDiv.textContent;
        };
        
        segDiv.appendChild(timeDiv);
        segDiv.appendChild(textDiv);
        transcriptionBody.appendChild(segDiv);
    });
    
    if (query === "") {
        transcriptionBody.scrollTop = transcriptionBody.scrollHeight;
    }
}

function renderSummary(points) {
    if (!points || points.length === 0) {
        summaryBody.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon text-purple-600"><i class="fa-solid fa-sparkles"></i></div>
                <h3>AI Summary</h3>
                <p>Summary will appear after transcription.</p>
            </div>
        `;
        return;
    }
    
    summaryBody.innerHTML = "";
    
    let html = '<div class="summary-card-block key-points"><div class="summary-block-header"><i class="fa-regular fa-lightbulb"></i><span>Key Points</span></div><ul class="summary-block-list">';
    points.forEach(point => {
        html += `<li class="summary-block-item">${point}</li>`;
    });
    html += '</ul></div>';
    
    summaryBody.innerHTML = html;
}

async function loadSessionHistory() {
    try {
        const currentUser = JSON.parse(localStorage.getItem('auraScribeCurrentUser'));
        const userId = currentUser ? currentUser.id : 'guest';
        
        const response = await fetch(`/api/sessions/${userId}`);
        const data = await response.json();
        
        if (data.success) {
            allSessionsList = data.sessions || [];
            renderSessions(allSessionsList);
        }
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

function renderSessions(sessions) {
    sessionsList.innerHTML = "";
    
    if (sessions.length === 0) {
        sessionsList.innerHTML = `
            <div class="empty-state" style="padding: 20px;">
                <i class="fa-regular fa-folder-open" style="font-size: 20px; opacity: 0.5;"></i>
                <p style="font-size: 11px;">No sessions yet.</p>
            </div>
        `;
        return;
    }
    
    sessions.forEach(session => {
        const item = document.createElement("div");
        item.className = `session-item ${appState.currentSessionId === session.session_id ? 'active' : ''}`;
        
        item.innerHTML = `
            <div class="session-header">
                <span class="session-date">${session.date}</span>
            </div>
            <span class="session-preview">${session.preview || 'No preview'}</span>
        `;
        
        item.onclick = () => loadHistoricalSession(session.session_id);
        sessionsList.appendChild(item);
    });
}

async function loadHistoricalSession(sessionId) {
    // Simple session loading
    appState.currentSessionId = sessionId;
    statusLabel.textContent = `Viewing: ${sessionId}`;
}

// ============================================================================
// BUTTON HANDLERS
// ============================================================================

recordBtn.onclick = () => {
    if (appState.isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
};

pauseBtn.onclick = () => togglePause();
stopBtn.onclick = () => stopRecording();

copyTextBtn.onclick = () => {
    const text = appState.transcribedSegments.map(s => s.text).join('\n');
    navigator.clipboard.writeText(text).then(() => {
        copyTextBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        setTimeout(() => {
            copyTextBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
        }, 2000);
    });
};

downloadDocxBtn.onclick = () => {
    if (appState.currentSessionId) {
        const currentUser = JSON.parse(localStorage.getItem('auraScribeCurrentUser'));
        const userId = currentUser ? currentUser.id : 'guest';
        window.location.href = `/api/download/docx/${appState.currentSessionId}/${userId}`;
    }
};

downloadPdfBtn.onclick = () => {
    if (appState.currentSessionId) {
        const currentUser = JSON.parse(localStorage.getItem('auraScribeCurrentUser'));
        const userId = currentUser ? currentUser.id : 'guest';
        window.location.href = `/api/download/pdf/${appState.currentSessionId}/${userId}`;
    }
};

clearTextBtn.onclick = () => {
    if (confirm("Clear all transcription?")) {
        appState.transcribedSegments = [];
        renderTranscription([]);
        renderSummary([]);
    }
};

refreshSessionsBtn.onclick = () => loadSessionHistory();

menuToggle.onclick = () => sidebar.classList.toggle('active');

document.addEventListener('click', (e) => {
    if (sidebar.classList.contains('active') && !sidebar.contains(e.target) && e.target !== menuToggle) {
        sidebar.classList.remove('active');
    }
});

transcriptionSearch.oninput = () => renderTranscription(appState.transcribedSegments);

historySearchInput.oninput = () => {
    const query = historySearchInput.value.toLowerCase();
    const filtered = allSessionsList.filter(s => 
        s.session_id.toLowerCase().includes(query) || 
        (s.preview && s.preview.toLowerCase().includes(query))
    );
    renderSessions(filtered);
};

// ============================================================================
// USER AUTHENTICATION
// ============================================================================

function getCurrentUserId() {
    const currentUser = JSON.parse(localStorage.getItem('auraScribeCurrentUser'));
    return currentUser ? currentUser.id : 'guest';
}

function checkUserLogin() {
    const currentUser = JSON.parse(localStorage.getItem('auraScribeCurrentUser'));
    const userInfo = document.getElementById('user-info');
    const guestInfo = document.getElementById('guest-info');
    
    if (currentUser) {
        if (userInfo) {
            userInfo.style.display = 'block';
            document.getElementById('user-name').textContent = currentUser.name;
            document.getElementById('user-email').textContent = currentUser.email;
        }
        if (guestInfo) guestInfo.style.display = 'none';
    } else {
        if (userInfo) userInfo.style.display = 'none';
        if (guestInfo) guestInfo.style.display = 'block';
    }
}

function logout() {
    localStorage.removeItem('auraScribeCurrentUser');
    checkUserLogin();
}

// ============================================================================
// INIT
// ============================================================================

window.onload = () => {
    loadSessionHistory();
    checkUserLogin();
    initSpeechRecognition();
};
