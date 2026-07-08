// WAV Audio Recorder Class (100% Client-Side Encoder)
class WavRecorder {
    constructor() {
        this.audioContext = null;
        this.processor = null;
        this.micStream = null;
        this.leftChannel = [];
        this.recordingLength = 0;
        this.sampleRate = 16000; // Target sample rate for Whisper
        this.analyser = null;
        this.lastProcessedIndex = 0;
        this.lastProcessedSamples = 0;
    }

    async start(analyserCallback) {
        this.leftChannel = [];
        this.recordingLength = 0;
        this.lastProcessedIndex = 0;
        this.lastProcessedSamples = 0;

        // Check for secure context and mediaDevices API
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Secure Context Required: Microphone access is blocked because the page is not loaded securely (use http://localhost:8000 or http://127.0.0.1:8000).");
        }

        try {
            // First attempt: high quality mono audio with noise cancellation
            this.micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    channelCount: 1
                },
                video: false
            });
        } catch (e) {
            console.warn("First microphone access attempt failed, trying fallback constraints...", e);
            try {
                // Second attempt: basic audio
                this.micStream = await navigator.mediaDevices.getUserMedia({
                    audio: true,
                    video: false
                });
            } catch (fallbackError) {
                // Third attempt: absolutely minimal request
                try {
                    this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                } catch (lastError) {
                    console.error("All microphone access attempts failed:", lastError);
                    throw lastError;
                }
            }
        }

        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioContextClass();
        
        const source = this.audioContext.createMediaStreamSource(this.micStream);
        const volume = this.audioContext.createGain();

        // Set up Analyser for visualization
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        source.connect(this.analyser);
        
        if (analyserCallback) {
            analyserCallback(this.analyser);
        }

        // Script Processor Node for direct sample collection
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        
        source.connect(volume);
        volume.connect(this.processor);
        this.processor.connect(this.audioContext.destination);

        this.processor.onaudioprocess = (e) => {
            const left = e.inputBuffer.getChannelData(0);
            // Clone the float samples to our buffer
            this.leftChannel.push(new Float32Array(left));
            this.recordingLength += 4096;
        };
    }

    pause() {
        if (this.audioContext) {
            this.audioContext.suspend();
        }
    }

    resume() {
        if (this.audioContext) {
            this.audioContext.resume();
        }
    }

    // Returns a copy of the WAV Blob accumulated SO FAR without stopping recording
    getWavBlobSoFar() {
        if (this.leftChannel.length === 0) return null;
        
        // Merge chunks
        const mergedBuffer = this.flattenBuffer(this.leftChannel, this.recordingLength);
        
        // Downsample
        const inputSampleRate = this.audioContext ? this.audioContext.sampleRate : 44100;
        const downsampledBuffer = this.downsampleBuffer(mergedBuffer, inputSampleRate, this.sampleRate);
        
        // Encode to WAV
        const wavBuffer = this.encodeWAV(downsampledBuffer);
        return new Blob([wavBuffer], { type: 'audio/wav' });
    }

    // Returns a copy of the WAV Blob accumulated SINCE the last processed chunk
    getNewWavChunk() {
        if (this.leftChannel.length <= this.lastProcessedIndex) return null;
        
        const inputSampleRate = this.audioContext ? this.audioContext.sampleRate : 44100;
        const chunkStartTime = this.lastProcessedSamples / inputSampleRate;
        
        const newBuffers = this.leftChannel.slice(this.lastProcessedIndex);
        let newLength = 0;
        for (let i = 0; i < newBuffers.length; i++) {
            newLength += newBuffers[i].length;
        }
        
        if (newLength === 0) return null;
        
        const mergedBuffer = this.flattenBuffer(newBuffers, newLength);
        const downsampledBuffer = this.downsampleBuffer(mergedBuffer, inputSampleRate, this.sampleRate);
        
        this.lastProcessedIndex = this.leftChannel.length;
        this.lastProcessedSamples += newLength;
        
        const wavBuffer = this.encodeWAV(downsampledBuffer);
        return {
            blob: new Blob([wavBuffer], { type: 'audio/wav' }),
            startTime: chunkStartTime
        };
    }

    stop() {
        if (this.processor) {
            this.processor.disconnect();
            this.processor.onaudioprocess = null;
        }
        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
        }
        
        const wavChunk = this.getNewWavChunk();
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        return wavChunk;
    }

    flattenBuffer(channelBuffer, recordingLength) {
        const result = new Float32Array(recordingLength);
        let offset = 0;
        for (let i = 0; i < channelBuffer.length; i++) {
            const buffer = channelBuffer[i];
            result.set(buffer, offset);
            offset += buffer.length;
        }
        return result;
    }

    downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
        if (inputSampleRate === outputSampleRate) {
            return buffer;
        }
        const sampleRateRatio = inputSampleRate / outputSampleRate;
        const newLength = Math.round(buffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        
        let offsetResult = 0;
        let offsetBuffer = 0;
        
        while (offsetResult < result.length) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
            let accum = 0, count = 0;
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
                count++;
            }
            result[offsetResult] = accum / count;
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }

    encodeWAV(samples) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        /* RIFF identifier */
        this.writeString(view, 0, 'RIFF');
        /* file length */
        view.setUint32(4, 36 + samples.length * 2, true);
        /* RIFF type */
        this.writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        this.writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw) */
        view.setUint16(20, 1, true);
        /* channel count */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, this.sampleRate, true);
        /* byte rate (sample rate * block align) */
        view.setUint32(28, this.sampleRate * 2, true);
        /* block align (channel count * bytes per sample) */
        view.setUint16(32, 2, true);
        /* bits per sample */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        this.writeString(view, 36, 'data');
        /* data chunk length */
        view.setUint32(40, samples.length * 2, true);
        
        this.floatTo16BitPCM(view, 44, samples);
        
        return view;
    }

    floatTo16BitPCM(output, offset, input) {
        for (let i = 0; i < input.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    }

    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
}

// App State Management
const appState = {
    isRecording: false,
    isPaused: false,
    recorder: null,
    timerInterval: null,
    autosaveInterval: null,
    poolRefreshInterval: null,
    secondsElapsed: 0,
    currentSessionId: null,
    transcribedSegments: [], // Array of {timestamp, text, start, end}
    analyserNode: null,
    animationFrameId: null,
    detectedLanguage: null // Locked language for the current recording session
};

let allSessionsList = []; // Global cache for sessions history list

// UI Elements
const recordBtn = document.getElementById('record-btn');
const pauseBtn = document.getElementById('pause-btn');
const stopBtn = document.getElementById('stop-btn');
const timerEl = document.getElementById('recording-timer');
const statusLabel = document.getElementById('status-label');
const statusDot = document.getElementById('status-indicator-dot');
const languageSelect = document.getElementById('language-select');
const poolStatusEl = document.getElementById('pool-status');
const continuousAutosave = document.getElementById('continuous-autosave');
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
const exportBtn = document.getElementById('export-btn');
const exportMenu = document.getElementById('export-menu');
const copySummaryBtn = document.getElementById('copy-summary-btn');
const regenerateSummaryBtn = document.getElementById('regenerate-summary-btn');

// Setup Audio Visualizer Canvas Size
function resizeCanvas() {
    waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
    waveformCanvas.height = waveformCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Initial Flat Line Drawing on Visualizer
function drawFlatLine() {
    canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    canvasCtx.fillStyle = '#FFFFFF';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    canvasCtx.lineWidth = 2.5;
    canvasCtx.strokeStyle = 'rgba(37, 99, 235, 0.35)'; // Accent blue glow
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, waveformCanvas.height / 2);
    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();
}
drawFlatLine();

// Live Audio Visualizer Loop (Redesigned for Premium Light Theme)
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
    
    // Clear and fill light canvas background
    canvasCtx.fillStyle = '#FFFFFF';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    
    // Draw subtle grid background lines
    canvasCtx.strokeStyle = 'rgba(0, 0, 0, 0.03)';
    canvasCtx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const y = (waveformCanvas.height / 4) * i;
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, y);
        canvasCtx.lineTo(waveformCanvas.width, y);
        canvasCtx.stroke();
    }
    
    canvasCtx.lineWidth = 2.5;
    
    // Create glowing blue-indigo gradient for wave curve
    const gradient = canvasCtx.createLinearGradient(0, 0, waveformCanvas.width, 0);
    gradient.addColorStop(0, '#2563EB'); // Blue accent
    gradient.addColorStop(0.5, '#8B5CF6'); // Purple accent
    gradient.addColorStop(1, '#2563EB'); // Blue accent
    
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

// Timer Formatting: MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Start Timer
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

// Stop Timer
function stopTimer() {
    clearInterval(appState.timerInterval);
    appState.timerInterval = null;
}

// Start Recording Event
async function startRecording() {
    try {
        resizeCanvas();
        appState.recorder = new WavRecorder();
        
        // Start recorder
        await appState.recorder.start((analyser) => {
            appState.analyserNode = analyser;
        });

        // Set session ID with timestamp
        const now = new Date();
        const dateStr = now.toISOString().slice(0, 10).replace(/-/g, '');
        const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '');
        appState.currentSessionId = `session_${dateStr}_${timeStr}`;

        // Reset state
        appState.isRecording = true;
        appState.isPaused = false;
        appState.transcribedSegments = [];
        appState.detectedLanguage = null;
        
        // UI updates
        recordBtn.classList.add('recording');
        recordBtn.title = "Recording... Click Stop to finish";
        
        pauseBtn.disabled = false;
        pauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>Pause</span>';
        stopBtn.disabled = false;
        
        // Disable settings adjustments during recording
        languageSelect.disabled = true;
        
        // Update Live Status Text & Dot Indicator to Recording state (Red)
        statusLabel.textContent = "Recording voice locally...";
        statusLabel.style.color = "var(--color-red)";
        statusDot.className = "status-indicator-dot active";
        statusDot.style.backgroundColor = "var(--color-red)";
        
        // Set document details
        transcriptionBody.innerHTML = `
            <div class="loading-placeholder">
                <i class="fa-solid fa-circle-notch fa-spin text-blue-600"></i>
                <p>Waiting for voice segments...</p>
            </div>
        `;
        summaryBody.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon text-purple-600"><i class="fa-solid fa-sparkles"></i></div>
                <h3>AI Summary Panel</h3>
                <p>Auto-generating summaries as you speak...</p>
            </div>
        `;
        
        // Start timers & visualization
        startTimer();
        visualize();
        startPoolStatusRefresh();
        
        // Set Continuous Autosave Interval (every 3 seconds for faster transcription)
        if (continuousAutosave.checked) {
            appState.autosaveInterval = setInterval(() => {
                if (!appState.isPaused) {
                    processAudioChunk(false);
                }
            }, 3000);
        }
        
        autosaveIndicator.style.visibility = "visible";
        autosaveIndicator.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin text-blue-600"></i> <span>Autosave engine monitoring...</span>';
        
    } catch (err) {
        console.error("Failed to access microphone or initialize AudioContext:", err);
        let errMsg = "Error: Cannot access microphone.";
        
        if (!window.isSecureContext) {
            errMsg = "Error: Non-Secure Connection. Please open via http://localhost:8000.";
        } else if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            errMsg = "Error: Microphone permission blocked. Allow mic access in browser.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            errMsg = "Error: No microphone hardware detected.";
        } else if (err.message && err.message.includes("Secure Context Required")) {
            errMsg = err.message;
        } else {
            errMsg = `Error: ${err.message || "Microphone access failed."}`;
        }
        
        statusLabel.textContent = errMsg;
        statusLabel.style.color = "var(--color-red)";
        statusDot.style.backgroundColor = "var(--color-red)";
    }
}

// Pause/Resume recording
function togglePause() {
    if (!appState.isRecording) return;
    
    if (appState.isPaused) {
        // Resume
        appState.recorder.resume();
        appState.isPaused = false;
        pauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>Pause</span>';
        
        statusLabel.textContent = "Recording voice locally...";
        statusLabel.style.color = "var(--color-red)";
        statusDot.className = "status-indicator-dot active";
        statusDot.style.backgroundColor = "var(--color-red)";
    } else {
        // Pause
        appState.recorder.pause();
        appState.isPaused = true;
        pauseBtn.innerHTML = '<i class="fa-solid fa-play"></i> <span>Resume</span>';
        
        statusLabel.textContent = "Recording paused.";
        statusLabel.style.color = "var(--color-amber)";
        statusDot.className = "status-indicator-dot";
        statusDot.style.backgroundColor = "var(--color-amber)";
        
        // Trigger a check-point save on pause
        processAudioChunk(true);
    }
}

// Stop Recording Event
async function stopRecording() {
    if (!appState.isRecording) return;
    
    statusLabel.textContent = "Finalizing transcription & generating summary...";
    statusLabel.style.color = "var(--accent-blue)";
    statusDot.className = "status-indicator-dot";
    statusDot.style.backgroundColor = "var(--accent-blue)";
    
    // Stop timers, intervals, visuals
    stopTimer();
    stopPoolStatusRefresh();
    clearInterval(appState.autosaveInterval);
    appState.autosaveInterval = null;
    cancelAnimationFrame(appState.animationFrameId);
    
    // Get final WAV chunk
    const finalWavChunk = appState.recorder.stop();
    appState.isRecording = false;
    appState.isPaused = false;
    
    // Reset UI buttons
    recordBtn.classList.remove('recording');
    recordBtn.title = "Start Recording";
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    
    drawFlatLine();
    
    if (finalWavChunk && finalWavChunk.blob) {
        await uploadAndTranscribeChunk(finalWavChunk.blob, finalWavChunk.startTime, true);
    } else {
        if (appState.transcribedSegments.length > 0) {
            await saveSession(true);
        } else {
            statusLabel.textContent = "Ready to record";
            statusLabel.style.color = "var(--text-secondary)";
            statusDot.style.backgroundColor = "var(--text-muted)";
            languageSelect.disabled = false;
        }
    }
}

// Process audio slice accumulated so far and execute background save
async function processAudioChunk(isQuietAutosave = false) {
    if (!appState.isRecording || !appState.recorder) return;
    
    const chunkData = appState.recorder.getNewWavChunk();
    if (chunkData) {
        if (!isQuietAutosave) {
            autosaveIndicator.innerHTML = '<i class="fa-solid fa-rotate fa-spin text-blue-600"></i> <span>Transcribing segment draft...</span>';
        }
        await uploadAndTranscribeChunk(chunkData.blob, chunkData.startTime, false);
    }
}

// Uploads Audio File to API and updates UI
async function uploadAndTranscribeChunk(audioBlob, startTime, isFinal = false) {
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");
    formData.append("language", appState.detectedLanguage || languageSelect.value);

    try {
        const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || "Transcription failed");
        }
        
        // If language auto-detection is active and speech is found, lock language for session
        if (data.detected_language && !appState.detectedLanguage && data.segments.length > 0) {
            appState.detectedLanguage = data.detected_language;
            console.log("Locked session language to:", appState.detectedLanguage);
        }
        
        // Adjust timestamps for the new segments based on startTime
        const adjustedSegments = data.segments.map(seg => {
            const startSec = startTime + seg.start;
            const endSec = startTime + seg.end;
            
            const startMin = Math.floor(startSec / 60);
            const startRemainingSec = Math.floor(startSec % 60);
            const endMin = Math.floor(endSec / 60);
            const endRemainingSec = Math.floor(endSec % 60);
            
            const timestamp = `${startMin.toString().padStart(2, '0')}:${startRemainingSec.toString().padStart(2, '0')} - ${endMin.toString().padStart(2, '0')}:${endRemainingSec.toString().padStart(2, '0')}`;
            
            return {
                timestamp: timestamp,
                text: seg.text,
                start: startSec,
                end: endSec
            };
        });
        
        // Append segments to state
        appState.transcribedSegments = appState.transcribedSegments.concat(adjustedSegments);
        
        // Update UI Transcription Panel
        renderTranscription(appState.transcribedSegments);
        
        // Trigger background autosave
        if (appState.transcribedSegments.length > 0) {
            await saveSession(isFinal);
        } else {
            if (isFinal) {
                transcriptionBody.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon"><i class="fa-solid fa-microphone-slash"></i></div>
                        <h3>No speech detected</h3>
                        <p>Whisper could not find voice segments in this recording session.</p>
                    </div>
                `;
                summaryBody.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon"><i class="fa-solid fa-circle-xmark"></i></div>
                        <h3>No summary points</h3>
                        <p>No summary could be generated.</p>
                    </div>
                `;
                statusLabel.textContent = "Ready to record";
                statusLabel.style.color = "var(--text-secondary)";
                statusDot.style.backgroundColor = "var(--text-muted)";
                modelSelect.disabled = false;
                languageSelect.disabled = false;
            }
        }
        
    } catch (err) {
        console.error("Transcription error:", err);
        autosaveIndicator.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-red-600"></i> <span class="text-red-600">Autosave failed: ${err.message}</span>`;
        if (isFinal) {
            statusLabel.textContent = "Ready to record";
            statusLabel.style.color = "var(--text-secondary)";
            statusDot.style.backgroundColor = "var(--text-muted)";
            languageSelect.disabled = false;
        }
    }
}

// Render transcribed segments into the transcription card
function renderTranscription(segments) {
    if (!segments || segments.length === 0) {
        transcriptionBody.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon"><i class="fa-solid fa-microphone-slash"></i></div>
                <h3>No transcription yet</h3>
                <p>Transcription segments with timestamps will show here as you record.</p>
            </div>
        `;
        return;
    }
    
    // Check if user is searching currently. If search input has value, preserve rendering with filter
    const query = transcriptionSearch.value.trim().toLowerCase();
    
    transcriptionBody.innerHTML = "";
    
    segments.forEach((seg, idx) => {
        const segDiv = document.createElement("div");
        segDiv.className = "transcription-segment";
        segDiv.dataset.index = idx;
        
        const timeDiv = document.createElement("div");
        timeDiv.className = "segment-time";
        timeDiv.textContent = seg.timestamp || "00:00 - 00:00";
        
        const textDiv = document.createElement("div");
        textDiv.className = "segment-text";
        textDiv.contentEditable = "true";
        
        // Handle search query highlights
        const text = seg.text;
        if (query !== "") {
            const matchIdx = text.toLowerCase().indexOf(query);
            if (matchIdx !== -1) {
                const originalText = text.substring(0, matchIdx);
                const matchingText = text.substring(matchIdx, matchIdx + query.length);
                const postText = text.substring(matchIdx + query.length);
                textDiv.innerHTML = `${originalText}<span class="search-highlight">${matchingText}</span>${postText}`;
            } else {
                textDiv.textContent = text;
                segDiv.style.opacity = "0.35"; // dim non-matching segment
            }
        } else {
            textDiv.textContent = text;
        }
        
        // Listen to edits so we can save changes back to appState.transcribedSegments
        textDiv.onblur = () => {
            appState.transcribedSegments[idx].text = textDiv.textContent;
            saveEditedSegment(idx, textDiv.textContent);
        };
        
        segDiv.appendChild(timeDiv);
        segDiv.appendChild(textDiv);
        transcriptionBody.appendChild(segDiv);
    });
    
    // Auto Scroll to bottom (only if not searching to prevent scroll disruption)
    if (query === "") {
        transcriptionBody.scrollTop = transcriptionBody.scrollHeight;
    }
}

// Background autosave helper after editing text inline
async function saveEditedSegment(index, newText) {
    if (index >= 0 && index < appState.transcribedSegments.length) {
        appState.transcribedSegments[index].text = newText;
        await saveSession(false);
    }
}

// Save active session data to Backend
async function saveSession(isFinal = false) {
    try {
        const response = await fetch("/api/save", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: appState.currentSessionId,
                text_segments: appState.transcribedSegments,
                language: languageSelect.value
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || "Save operation failed");
        }
        
        // Update Autosave footer indicator
        const timeNow = new Date().toLocaleTimeString();
        autosaveIndicator.style.visibility = "visible";
        autosaveIndicator.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span id="autosave-indicator-text">Draft saved locally at ${timeNow}</span>`;
        
        // Enable file download buttons
        downloadDocxBtn.disabled = false;
        
        // Update PDF summary if summary is returned
        if (data.summary_points && data.summary_points.length > 0) {
            renderSummary(data.summary_points);
            downloadPdfBtn.disabled = false;
        }
        
        if (isFinal) {
            statusLabel.textContent = "Transcription session complete. Files saved!";
            statusLabel.style.color = "var(--color-green)";
            statusDot.className = "status-indicator-dot";
            statusDot.style.backgroundColor = "var(--color-green)";
            
            // Re-enable options
            languageSelect.disabled = false;
            
            // Refresh history drawer
            loadSessionHistory();
        }
        
    } catch (err) {
        console.error("Save error:", err);
        autosaveIndicator.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-red-600"></i> <span class="text-red-600">Autosave failed: ${err.message}</span>`;
    }
}

// Render Summary Points with SaaS Category Blocks (Notion/Linear style)
function renderSummary(points) {
    if (!points || points.length === 0) {
        summaryBody.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon text-purple-600"><i class="fa-solid fa-sparkles"></i></div>
                <h3>AI Summary Panel</h3>
                <p>Key points, action items, important decisions, and highlights will populate once you stop recording or load history.</p>
            </div>
        `;
        return;
    }
    
    summaryBody.innerHTML = "";
    
    // Classify summary points into categories
    const keyPoints = [];
    const actionItems = [];
    const decisions = [];
    const highlights = [];
    
    points.forEach(point => {
        const lower = point.toLowerCase();
        if (lower.includes("need") || lower.includes("should") || lower.includes("will") || lower.includes("todo") || lower.includes("action") || lower.includes("task") || lower.includes("assign")) {
            actionItems.push(point);
        } else if (lower.includes("decid") || lower.includes("agree") || lower.includes("conclud") || lower.includes("resolv") || lower.includes("chose") || lower.includes("determination")) {
            decisions.push(point);
        } else if (lower.includes("import") || lower.includes("key") || lower.includes("note") || lower.includes("highlight") || lower.includes("critical") || lower.includes("essential")) {
            highlights.push(point);
        } else {
            keyPoints.push(point);
        }
    });
    
    // Fallback: If points are not classified because of PageRank phrasing, distribute them evenly across categories
    if (actionItems.length === 0 && decisions.length === 0 && highlights.length === 0) {
        points.forEach((point, idx) => {
            if (idx % 4 === 0) keyPoints.push(point);
            else if (idx % 4 === 1) actionItems.push(point);
            else if (idx % 4 === 2) decisions.push(point);
            else highlights.push(point);
        });
        
        // Avoid duplicate in keyPoints
        keyPoints.length = 0;
        points.forEach((point, idx) => {
            if (idx % 4 === 0) keyPoints.push(point);
        });
    }
    
    // Helper to generate categories card block
    function createCategoryBlock(title, iconClass, className, items) {
        if (!items || items.length === 0) return "";
        
        let listHtml = "";
        items.forEach(item => {
            listHtml += `<li class="summary-block-item">${item}</li>`;
        });
        
        return `
            <div class="summary-card-block ${className}">
                <div class="summary-block-header">
                    <i class="${iconClass}"></i>
                    <span>${title}</span>
                </div>
                <ul class="summary-block-list">
                    ${listHtml}
                </ul>
            </div>
        `;
    }
    
    let blocksHtml = "";
    blocksHtml += createCategoryBlock("Key Points", "fa-regular fa-lightbulb", "key-points", keyPoints);
    blocksHtml += createCategoryBlock("Action Items", "fa-regular fa-circle-check", "action-items", actionItems);
    blocksHtml += createCategoryBlock("Important Decisions", "fa-solid fa-gavel", "decisions", decisions);
    blocksHtml += createCategoryBlock("Highlights", "fa-regular fa-star", "highlights", highlights);
    
    summaryBody.innerHTML = blocksHtml;
    
    // Add Exports link footer
    const previewDiv = document.createElement("div");
    previewDiv.className = "summary-preview-section";
    previewDiv.innerHTML = `
        <h4 class="summary-preview-heading">Session Exports</h4>
        <div class="session-downloads-row">
            <a href="/api/download/docx/${appState.currentSessionId}" class="download-link-btn" id="open-word-link">
                <i class="fa-regular fa-file-word text-blue-600"></i> Open Word
            </a>
            <a href="/api/download/pdf/${appState.currentSessionId}" class="download-link-btn" id="open-pdf-link">
                <i class="fa-regular fa-file-pdf text-red-600"></i> Open PDF Summary
            </a>
        </div>
    `;
    summaryBody.appendChild(previewDiv);
}

// Fetch session list from Backend
async function loadSessionHistory() {
    try {
        const response = await fetch("/api/sessions");
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error);
        }
        
        allSessionsList = data.sessions || [];
        renderSessions(allSessionsList);
    } catch (err) {
        console.error("Failed to load history:", err);
        sessionsList.innerHTML = `
            <div style="padding: 10px; color: var(--color-red); font-size:11px; text-align:center;">
                <i class="fa-solid fa-circle-exclamation"></i> Error loading history
            </div>
        `;
    }
}

// Render historical session items into the sidebar history list
function renderSessions(sessions) {
    sessionsList.innerHTML = "";
    
    if (sessions.length === 0) {
        sessionsList.innerHTML = `
            <div class="empty-state" style="padding: 20px;">
                <i class="fa-regular fa-folder-open" style="font-size: 20px; opacity: 0.5;"></i>
                <p style="font-size: 11px;">No sessions found.</p>
            </div>
        `;
        return;
    }
    
    sessions.forEach(session => {
        const item = document.createElement("div");
        item.className = `session-item ${appState.currentSessionId === session.session_id ? 'active' : ''}`;
        
        const header = document.createElement("div");
        header.className = "session-header";
        
        const dateSpan = document.createElement("span");
        dateSpan.className = "session-date";
        dateSpan.textContent = session.date;
        
        const actions = document.createElement("div");
        actions.className = "session-actions";
        
        const docxLink = document.createElement("button");
        docxLink.className = "btn-docx";
        docxLink.innerHTML = '<i class="fa-regular fa-file-word"></i>';
        docxLink.title = "Download Word file";
        docxLink.onclick = (e) => {
            e.stopPropagation();
            window.location.href = `/api/download/docx/${session.session_id}`;
        };
        actions.appendChild(docxLink);
        
        if (session.pdf_exists) {
            const pdfLink = document.createElement("button");
            pdfLink.className = "btn-pdf";
            pdfLink.innerHTML = '<i class="fa-regular fa-file-pdf"></i>';
            pdfLink.title = "Download PDF Summary";
            pdfLink.onclick = (e) => {
                e.stopPropagation();
                window.location.href = `/api/download/pdf/${session.session_id}`;
            };
            actions.appendChild(pdfLink);
        }
        
        header.appendChild(dateSpan);
        header.appendChild(actions);
        
        const preview = document.createElement("span");
        preview.className = "session-preview";
        preview.textContent = session.preview || "[Empty Session]";
        
        item.appendChild(header);
        item.appendChild(preview);
        
        item.onclick = () => loadHistoricalSession(session.session_id);
        
        sessionsList.appendChild(item);
    });
}

// Load historical session content from disk to UI
async function loadHistoricalSession(sessionId) {
    if (appState.isRecording) {
        if (!confirm("Are you sure you want to open this session? Active recording will be stopped.")) {
            return;
        }
        await stopRecording();
    }
    
    statusLabel.textContent = `Viewing historical session: ${sessionId}`;
    statusLabel.style.color = "var(--text-secondary)";
    statusDot.className = "status-indicator-dot";
    statusDot.style.backgroundColor = "var(--text-secondary)";
    
    appState.currentSessionId = sessionId;
    
    // Highlight item in sidebar list
    const items = sessionsList.querySelectorAll('.session-item');
    items.forEach(el => el.classList.remove('active'));
    
    // Find matching item in sidebar and activate it
    const sidebarItems = Array.from(items);
    const matchedItem = sidebarItems.find(el => el.textContent.includes(sessionId.replace('session_', '')));
    if (matchedItem) matchedItem.classList.add('active');
    
    // Set UI loading placeholder
    transcriptionBody.innerHTML = `
        <div class="loading-placeholder">
            <i class="fa-solid fa-circle-notch fa-spin text-blue-600"></i>
            <p>Loading transcription file...</p>
        </div>
    `;
    summaryBody.innerHTML = `
        <div class="loading-placeholder">
            <i class="fa-solid fa-circle-notch fa-spin text-purple-600"></i>
            <p>Loading summaries...</p>
        </div>
    `;
    
    try {
        const docxResponse = await fetch(`/api/download/docx/${sessionId}`);
        if (!docxResponse.ok) {
            throw new Error("Could not find session documents on disk.");
        }
        
        // Since the backend saves a txt draft, let's load it to show the full text
        const txtResponse = await fetch(`/transcriptions/${sessionId}_draft.txt`);
        let draftText = "";
        if (txtResponse.ok) {
            draftText = await txtResponse.text();
        }
        
        // Parse segments from txt file
        transcriptionBody.innerHTML = "";
        const lines = draftText.split('\n');
        const parsedSegments = [];
        
        if (lines.length > 0 && draftText.trim() !== "") {
            lines.forEach((line) => {
                if (line.trim() === "") return;
                
                // Match the timestamp format [MM:SS - MM:SS] at start of the line
                const match = line.match(/^\[([\d:\s-]+)\]\s*(.*)$/);
                if (match) {
                    const timestamp = match[1].trim();
                    const text = match[2].trim();
                    
                    parsedSegments.push({
                        timestamp: timestamp,
                        text: text
                    });
                } else {
                    parsedSegments.push({
                        timestamp: "00:00 - 00:00",
                        text: line.trim()
                    });
                }
            });
        }
        
        appState.transcribedSegments = parsedSegments;
        renderTranscription(parsedSegments);
        
        // Run summarization to populate summary panel in the UI dynamically
        if (draftText.trim().length > 30 && parsedSegments.length > 0) {
            const response = await fetch("/api/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    text_segments: parsedSegments,
                    language: "auto"
                })
            });
            const data = await response.json();
            if (data.success) {
                renderSummary(data.summary_points);
            }
        } else {
            summaryBody.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon text-amber-600"><i class="fa-solid fa-triangle-exclamation"></i></div>
                    <h3>Summary Unavailable</h3>
                    <p>No summary points could be generated for short text.</p>
                </div>
            `;
        }
        
        // Enable buttons
        downloadDocxBtn.disabled = false;
        downloadPdfBtn.disabled = false;
        autosaveIndicator.style.visibility = "visible";
        autosaveIndicator.innerHTML = '<i class="fa-solid fa-circle-check text-green-600"></i> <span>Displaying saved session</span>';
        
    } catch (err) {
        console.error("Failed to load historical session:", err);
        transcriptionBody.innerHTML = `<div class="empty-state"><div class="empty-state-icon text-red-600"><i class="fa-solid fa-triangle-exclamation"></i></div><h3>Failed to load</h3><p>Could not load session content.</p></div>`;
        summaryBody.innerHTML = `<div class="empty-state"><div class="empty-state-icon text-red-600"><i class="fa-solid fa-triangle-exclamation"></i></div><h3>Failed to load</h3><p>Could not load summary file.</p></div>`;
    }
}

// History drawer search filter action
historySearchInput.oninput = () => {
    const query = historySearchInput.value.trim().toLowerCase();
    const filtered = allSessionsList.filter(session => {
        return session.session_id.toLowerCase().includes(query) ||
               (session.preview && session.preview.toLowerCase().includes(query)) ||
               session.date.toLowerCase().includes(query);
    });
    renderSessions(filtered);
};

// Transcription text search filter action
transcriptionSearch.oninput = () => {
    // Re-render segments with matches highlighted
    renderTranscription(appState.transcribedSegments);
};

// Clear Transcription Text Action
clearTextBtn.onclick = () => {
    if (appState.transcribedSegments.length === 0) {
        alert("Nothing to clear!");
        return;
    }
    if (confirm("Are you sure you want to clear the active transcription? This will empty the editor and summary, but won't delete files on disk unless you save again.")) {
        appState.transcribedSegments = [];
        renderTranscription([]);
        renderSummary([]);
        downloadDocxBtn.disabled = true;
        downloadPdfBtn.disabled = true;
        autosaveIndicator.style.visibility = "hidden";
        transcriptionSearch.value = "";
    }
};

// Export Dropdown menu toggler
exportBtn.onclick = (e) => {
    e.stopPropagation();
    exportMenu.classList.toggle('active');
};
document.addEventListener('click', () => {
    exportMenu.classList.remove('active');
});

// Export triggers
document.getElementById('export-docx').onclick = () => {
    if (appState.currentSessionId) {
        window.location.href = `/api/download/docx/${appState.currentSessionId}`;
    } else {
        alert("No active session to export!");
    }
};

document.getElementById('export-pdf').onclick = () => {
    if (appState.currentSessionId) {
        window.location.href = `/api/download/pdf/${appState.currentSessionId}`;
    } else {
        alert("No active session to export!");
    }
};

document.getElementById('export-txt').onclick = () => {
    if (appState.transcribedSegments.length === 0) {
        alert("No transcription text to export!");
        return;
    }
    const textLines = appState.transcribedSegments.map(seg => `[${seg.timestamp}] ${seg.text}`);
    const fullText = textLines.join('\n');
    const blob = new Blob([fullText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${appState.currentSessionId || 'transcription'}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

// Copy Summary text to Clipboard
copySummaryBtn.onclick = () => {
    const listItems = Array.from(summaryBody.querySelectorAll('.summary-block-item')).map(el => el.textContent);
    const fullSummaryText = listItems.join('\n');
    if (fullSummaryText.trim() === "") {
        alert("Nothing to copy yet!");
        return;
    }
    
    navigator.clipboard.writeText(fullSummaryText).then(() => {
        const origHtml = copySummaryBtn.innerHTML;
        copySummaryBtn.innerHTML = '<i class="fa-solid fa-check text-green-600"></i> <span>Copied!</span>';
        setTimeout(() => {
            copySummaryBtn.innerHTML = origHtml;
        }, 2000);
    }).catch(err => {
        console.error("Summary copy failed:", err);
    });
};

// Regenerate summary points
regenerateSummaryBtn.onclick = async () => {
    if (appState.transcribedSegments.length === 0) {
        alert("Cannot generate summary: transcription is empty.");
        return;
    }
    
    regenerateSummaryBtn.disabled = true;
    const origHtml = regenerateSummaryBtn.innerHTML;
    regenerateSummaryBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin text-blue-600"></i> <span>Regenerating...</span>';
    
    try {
        await saveSession(false);
    } catch (err) {
        console.error("Regeneration failed:", err);
        alert(`Failed to regenerate: ${err.message}`);
    } finally {
        regenerateSummaryBtn.disabled = false;
        regenerateSummaryBtn.innerHTML = origHtml;
    }
};

// Button click bindings
recordBtn.onclick = () => {
    if (appState.isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
};

pauseBtn.onclick = () => {
    togglePause();
};

stopBtn.onclick = () => {
    stopRecording();
};

copyTextBtn.onclick = () => {
    const textSegments = Array.from(transcriptionBody.querySelectorAll('.segment-text')).map(el => el.textContent);
    const fullText = textSegments.join('\n') || transcriptionBody.textContent;
    if (fullText.trim() === "" || fullText.includes("No transcription yet")) {
        alert("Nothing to copy yet!");
        return;
    }
    
    navigator.clipboard.writeText(fullText).then(() => {
        const origHtml = copyTextBtn.innerHTML;
        copyTextBtn.innerHTML = '<i class="fa-solid fa-check text-green-600"></i> <span>Copied!</span>';
        setTimeout(() => {
            copyTextBtn.innerHTML = origHtml;
        }, 2000);
    }).catch(err => {
        console.error("Clipboard copy failed:", err);
    });
};

downloadDocxBtn.onclick = () => {
    if (appState.currentSessionId) {
        window.location.href = `/api/download/docx/${appState.currentSessionId}`;
    }
};

downloadPdfBtn.onclick = () => {
    if (appState.currentSessionId) {
        window.location.href = `/api/download/pdf/${appState.currentSessionId}`;
    }
};

refreshSessionsBtn.onclick = () => {
    loadSessionHistory();
};

menuToggle.onclick = (e) => {
    e.stopPropagation();
    sidebar.classList.toggle('active');
};

document.addEventListener('click', (e) => {
    if (sidebar.classList.contains('active') && !sidebar.contains(e.target) && e.target !== menuToggle) {
        sidebar.classList.remove('active');
    }
});

// App Startup Code
window.onload = () => {
    loadSessionHistory();
    loadPoolStatus();
};

// Fetch and display model pool status
async function loadPoolStatus() {
    try {
        const response = await fetch("/api/pool-status");
        const data = await response.json();
        
        if (data.success && data.status) {
            const status = data.status;
            const modelCount = status.total_models;
            
            if (modelCount > 0) {
                // Show loaded models count and list
                const voskCount = status.models.filter(m => m.startsWith('vosk')).length;
                const whisperCount = status.models.filter(m => m.startsWith('whisper')).length;
                
                poolStatusEl.innerHTML = `
                    <div class="pool-ready">
                        <i class="fa-solid fa-check-circle text-green-600"></i>
                        <span><strong>${modelCount}</strong> Models Ready</span>
                    </div>
                    <div class="pool-breakdown">
                        <span class="pool-tag vosk-tag"><i class="fa-solid fa-bolt"></i> ${voskCount} Vosk (Fast)</span>
                        <span class="pool-tag whisper-tag"><i class="fa-solid fa-brain"></i> ${whisperCount} Whisper (Accurate)</span>
                    </div>
                `;
            } else {
                // Models still loading
                poolStatusEl.innerHTML = `
                    <div class="pool-loading">
                        <i class="fa-solid fa-circle-notch fa-spin"></i>
                        <span>Loading models...</span>
                    </div>
                `;
                // Retry in 3 seconds
                setTimeout(loadPoolStatus, 3000);
            }
        }
    } catch (err) {
        console.error("Failed to load pool status:", err);
        poolStatusEl.innerHTML = `
            <div class="pool-error">
                <i class="fa-solid fa-triangle-exclamation text-amber-600"></i>
                <span>Checking status...</span>
            </div>
        `;
        // Retry in 5 seconds
        setTimeout(loadPoolStatus, 5000);
    }
}

// Refresh pool status periodically during recording
function startPoolStatusRefresh() {
    appState.poolRefreshInterval = setInterval(loadPoolStatus, 10000);
}

function stopPoolStatusRefresh() {
    if (appState.poolRefreshInterval) {
        clearInterval(appState.poolRefreshInterval);
        appState.poolRefreshInterval = null;
    }
}

// ==================== USER AUTHENTICATION ====================

function checkUserLogin() {
    const currentUser = JSON.parse(localStorage.getItem('auraScribeCurrentUser'));
    const userInfo = document.getElementById('user-info');
    const guestInfo = document.getElementById('guest-info');
    const headerUserInfo = document.getElementById('header-user-info');
    
    if (currentUser) {
        userInfo.style.display = 'block';
        guestInfo.style.display = 'none';
        document.getElementById('user-name').textContent = currentUser.name;
        document.getElementById('user-email').textContent = currentUser.email;
        
        // Show header user info
        if (headerUserInfo) {
            headerUserInfo.style.display = 'flex';
            document.getElementById('header-user-name').textContent = currentUser.name;
        }
    } else {
        userInfo.style.display = 'none';
        guestInfo.style.display = 'block';
        if (headerUserInfo) {
            headerUserInfo.style.display = 'none';
        }
    }
}

function logout() {
    localStorage.removeItem('auraScribeCurrentUser');
    checkUserLogin();
}

// Check login on page load
checkUserLogin();
