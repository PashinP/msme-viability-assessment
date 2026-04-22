"""
Voice Input Component for Streamlit
=====================================
Uses the browser's Web Speech API (SpeechRecognition) to capture
voice input and return it as text to the Streamlit app.

Works in Chrome, Edge, and Safari. Firefox has limited support.
"""

import streamlit.components.v1 as components


def voice_input_button(key: str = "voice_input", height: int = 60) -> str | None:
    """
    Renders a microphone button that captures voice input.
    Returns the transcribed text, or None if nothing captured.
    """
    result = components.html(
        _VOICE_HTML,
        height=height,
        key=key,
    )
    return result


_VOICE_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Inter', -apple-system, sans-serif;
        background: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .voice-container {
        display: flex;
        align-items: center;
        gap: 12px;
        width: 100%;
    }
    .mic-btn {
        width: 44px; height: 44px;
        border-radius: 50%;
        border: none;
        background: linear-gradient(135deg, #302b63, #24243e);
        color: white;
        font-size: 20px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        flex-shrink: 0;
    }
    .mic-btn:hover {
        transform: scale(1.08);
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .mic-btn.listening {
        background: linear-gradient(135deg, #d32f2f, #f44336);
        animation: pulse 1.5s ease infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(244,67,54,0.4); }
        50% { box-shadow: 0 0 0 12px rgba(244,67,54,0); }
    }
    .status-text {
        font-size: 13px;
        color: #888;
        flex-grow: 1;
    }
    .status-text.active { color: #f44336; font-weight: 500; }
    .status-text.done { color: #4caf50; }
    .status-text.error { color: #ff9800; }
</style>
</head>
<body>
<div class="voice-container">
    <button class="mic-btn" id="micBtn" onclick="toggleListening()">🎤</button>
    <span class="status-text" id="statusText">Click mic to speak</span>
</div>

<script>
    let recognition = null;
    let isListening = false;
    let finalTranscript = '';

    const micBtn = document.getElementById('micBtn');
    const statusText = document.getElementById('statusText');

    // Check browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        statusText.textContent = '⚠️ Voice not supported in this browser. Use Chrome.';
        statusText.className = 'status-text error';
        micBtn.style.opacity = '0.5';
        micBtn.style.cursor = 'not-allowed';
    } else {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'hi-IN';  // Hindi + English mixed
        // Also accept English
        recognition.lang = 'en-IN';  // Indian English (understands Hindi words too)

        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add('listening');
            statusText.textContent = '🔴 Listening... speak now';
            statusText.className = 'status-text active';
            finalTranscript = '';
        };

        recognition.onresult = (event) => {
            let interim = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript + ' ';
                } else {
                    interim += event.results[i][0].transcript;
                }
            }
            if (finalTranscript.trim()) {
                statusText.textContent = '✅ ' + finalTranscript.trim();
                statusText.className = 'status-text done';
            } else if (interim) {
                statusText.textContent = '🎙️ ' + interim;
                statusText.className = 'status-text active';
            }
        };

        recognition.onend = () => {
            isListening = false;
            micBtn.classList.remove('listening');
            if (finalTranscript.trim()) {
                statusText.textContent = '✅ ' + finalTranscript.trim();
                statusText.className = 'status-text done';
                // Send result to Streamlit
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: finalTranscript.trim()
                }, '*');
            } else {
                statusText.textContent = 'Click mic to speak';
                statusText.className = 'status-text';
            }
        };

        recognition.onerror = (event) => {
            isListening = false;
            micBtn.classList.remove('listening');
            if (event.error === 'not-allowed') {
                statusText.textContent = '⚠️ Mic access denied. Allow in browser settings.';
            } else if (event.error === 'no-speech') {
                statusText.textContent = 'No speech detected. Try again.';
            } else {
                statusText.textContent = '⚠️ Error: ' + event.error;
            }
            statusText.className = 'status-text error';
        };
    }

    function toggleListening() {
        if (!recognition) return;
        if (isListening) {
            recognition.stop();
        } else {
            recognition.start();
        }
    }
</script>
</body>
</html>
"""
