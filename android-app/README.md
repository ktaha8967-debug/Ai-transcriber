# AuraScribe Pro - Android APK Build Guide

## GitHub Actions Se APK Kaise Banaye (Sabse Easy!)

### Step 1: GitHub Pe Repository Banao
1. GitHub.com pe jao
2. Naya repository banao (public ya private)
3. Name: `voice-transcriber`

### Step 2: Code Push Karo
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/voice-transcriber.git
git push -u origin main
```

### Step 3: APK Build Karo
1. GitHub repository pe jao
2. **Actions** tab pe click karo
3. **Build AuraScribe Pro APK** workflow dikhega
4. **Run workflow** pe click karo
5. **Run workflow** button dabao

### Step 4: APK Download Karo
1. Build complete hone do (5-10 min)
2. **Actions** tab pe jao
3. Completed workflow pe click karo
4. **Artifacts** section mein `AuraScribe-Pro-APK` dikhega
5. Download karo!

---

## APK Install Kaise Kare (Phone Pe)

1. APK file phone mein transfer karo
2. **Settings > Security > Unknown Sources** enable karo
3. APK file open karo
4. Install karo
5. Open karo!

---

## Important Notes

### Server Kahan Chalega?
- **Option 1:** PC pe server chalao, phone se WiFi pe access karo
- **Option 2:** Phone pe Termux mein server chalao

### Server URL Kaise Set Kare?
`android-app/app/src/main/java/com/aurascribe/pro/MainActivity.java` mein:

```java
// PC ke liye (same WiFi network):
private static final String SERVER_URL = "http://192.168.1.xxx:8000";

// Phone ke liye (Termux server):
private static final String SERVER_URL = "http://localhost:8000";
```

### WiFi Se Kaise Connect Kare?
1. PC pe `ipconfig` run karo
2. IPv4 address dekho (e.g., 192.168.1.100)
3. Phone browser mein: `http://192.168.1.100:8000`

---

## Features
- Full voice transcription
- 11 AI models (Vosk + Whisper)
- Auto-distribution across models
- 3-second batch processing
- Word/PDF export
- Session history
- 100% offline capable

---

## Troubleshooting

### "Build Failed" Error
- GitHub Actions log check karo
- Error message dekho
- Code mein koi syntax error ho sakta hai

### APK Install Nahi Ho Raha
- "Unknown Sources" enable karo
- Phone storage check karo
- APK file corrupt ho sakta hai - dobara download karo

### Server Se Connect Nahi Ho Raha
- PC pe server chal raha hai check karo
- Same WiFi network pe ho
- Firewall block nahi kar raha
- IP address sahi hai

---

## Support
Issue ho to GitHub repository pe issue create karo!
