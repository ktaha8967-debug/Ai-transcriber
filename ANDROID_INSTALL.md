# AuraScribe Pro - Android Installation Guide

## Option 1: Termux Method (RECOMMENDED - Easiest)

### Step 1: Install Termux
Download and install **Termux** from F-Droid (NOT Play Store):
- https://f-droid.org/en/packages/com.termux/

### Step 2: Setup Termux
Open Termux and run these commands:

```bash
# Update Termux
pkg update -y && pkg upgrade -y

# Install Python and dependencies
pkg install -y python clang libffi openssl libcrypt binutils rust git

# Install Python packages
pip install --upgrade pip
pip install fastapi uvicorn faster-whisper vosk httpx python-docx fpdf2 pydantic
```

### Step 3: Copy App Files
You need to copy these files to your phone:
- `app.py`
- `summarizer.py`
- `document_generator.py`
- `static/` folder (with index.html, script.js, style.css)
- `.env` file

**Easy ways to copy files:**
1. Use USB cable to connect phone to PC
2. Use Termux file manager: `pkg install termux-file-manager`
3. Use GitHub if your code is there

### Step 4: Run the Server
```bash
cd ~/aura-scribe
python app.py
```

### Step 5: Open in Browser
Open any browser on your phone and go to:
```
http://localhost:8000
```

**That's it!** The app is now running on your phone.

---

## Option 2: Build APK (Advanced)

### Prerequisites
1. Install Android Studio: https://developer.android.com/studio
2. Install Java JDK 17

### Step 1: Open Project
1. Open Android Studio
2. Click "Open an Existing Project"
3. Navigate to `android-app` folder
4. Wait for Gradle sync to complete

### Step 2: Configure Server URL
Edit `app/src/main/java/com/aurascribe/pro/MainActivity.java`:

```java
// For phone running server on same device:
private static final String SERVER_URL = "http://10.0.2.2:8000";

// For phone connecting to PC server:
// Replace YOUR_PC_IP with your PC's IP address
private static final String SERVER_URL = "http://192.168.1.xxx:8000";
```

### Step 3: Build APK
1. Click "Build" menu
2. Select "Build Bundle(s) / APK(s)"
3. Click "Build APK(s)"
4. Wait for build to complete
5. APK will be in `app/build/outputs/apk/debug/`

### Step 4: Install on Phone
1. Transfer APK to your phone
2. Open the APK file
3. Enable "Install from Unknown Sources" if asked
4. Install and open the app

---

## Finding Your PC's IP Address

If you want to access the server from your phone's browser:

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.100)

**Mac/Linux:**
```bash
ifconfig
```
or
```bash
hostname -I
```

---

## Troubleshooting

### "Connection Refused" Error
- Make sure the Python server is running
- Check that phone and PC are on same WiFi network
- Try using PC's IP address instead of localhost

### Microphone Not Working
- Grant microphone permission when prompted
- Check phone's Settings > Apps > Browser > Permissions

### Models Not Loading
- First run downloads models automatically
- Make sure you have internet connection for first run
- Models are cached after first download

### App Running Slow
- Close other apps to free up RAM
- Use "Vosk" models for faster transcription
- Reduce batch size if needed

---

## Quick Start (Termux - Copy & Paste)

```bash
# Install everything
pkg update -y && pkg upgrade -y
pkg install -y python clang libffi openssl libcrypt binutils rust git
pip install --upgrade pip
pip install fastapi uvicorn faster-whisper vosk httpx python-docx fpdf2 pydantic

# Copy your app files to ~/aura-scribe folder, then:
cd ~/aura-scribe
python app.py
```

Open browser: `http://localhost:8000`

---

## Features on Android

- Full voice transcription with 11 AI models
- Auto-distribution across models
- 3-second batch processing
- Word document export
- PDF summary export
- Session history
- Works offline (after first model download)

---

## Requirements

- Android 7.0 (API 24) or higher
- 2GB RAM recommended
- 500MB free storage (for models)
- Microphone permission

---

## Support

If you encounter issues:
1. Check this guide first
2. Make sure all steps are followed
3. Check Termux output for error messages
4. Ensure stable internet for first setup
