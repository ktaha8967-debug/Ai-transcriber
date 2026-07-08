#!/bin/bash
# ========================================
# AuraScribe Pro - Termux Setup Script
# Phone pe Python server chalane ke liye
# ========================================

echo "========================================="
echo "   AuraScribe Pro - Termux Setup"
echo "========================================="
echo ""

# Step 1: Update Termux
echo "[1/8] Updating Termux packages..."
pkg update -y
pkg upgrade -y

# Step 2: Install required packages
echo "[2/8] Installing required packages..."
pkg install -y python
pkg install -y clang
pkg install -y libffi
pkg install -y openssl
pkg install -y libcrypt
pkg install -y binutils
pkg install -y rust
pkg install -y git

# Step 3: Install Python dependencies
echo "[3/8] Installing Python packages..."
pip install --upgrade pip
pip install fastapi
pip install uvicorn
pip install faster-whisper
pip install vosk
pip install httpx
pip install python-docx
pip install fpdf2
pip install pydantic

# Step 4: Create project directory
echo "[4/8] Creating project directory..."
mkdir -p ~/aura-scribe
cd ~/aura-scribe

# Step 5: Copy app files (user needs to do this manually or use git)
echo "[5/8] Setting up app files..."
echo ""
echo "IMPORTANT: You need to copy these files to ~/aura-scribe/"
echo ""
echo "Files to copy:"
echo "  - app.py"
echo "  - summarizer.py"
echo "  - document_generator.py"
echo "  - static/ folder (index.html, script.js, style.css)"
echo "  - .env file (with your API keys)"
echo ""
echo "You can use:"
echo "  1. USB file transfer"
echo "  2. Git clone (if on GitHub)"
echo "  3. Termux file manager"
echo ""

# Step 6: Create start script
echo "[6/8] Creating start script..."
cat > start.sh << 'EOF'
#!/bin/bash
cd ~/aura-scribe
echo "Starting AuraScribe Pro server..."
echo ""
echo "Open your browser and go to:"
echo "  http://localhost:8000"
echo ""
echo "Or from another device on same WiFi:"
echo "  http://$(hostname -I | awk '{print $1}'):8000"
echo ""
python app.py
EOF
chmod +x start.sh

# Step 7: Create auto-start script
echo "[7/8] Creating auto-start script..."
cat > autostart.sh << 'EOF'
#!/bin/bash
# Auto-start server when Termux opens
cd ~/aura-scribe
python app.py &
sleep 2
echo ""
echo "Server started! Open browser to http://localhost:8000"
EOF
chmod +x autostart.sh

# Step 8: Final instructions
echo "[8/8] Setup complete!"
echo ""
echo "========================================="
echo "   Setup Complete!"
echo "========================================="
echo ""
echo "To start the server:"
echo "  cd ~/aura-scribe"
echo "  bash start.sh"
echo ""
echo "Or use the auto-start script:"
echo "  bash autostart.sh"
echo ""
echo "Then open browser and go to:"
echo "  http://localhost:8000"
echo ""
echo "========================================="
echo "   IMPORTANT NOTES:"
echo "========================================="
echo ""
echo "1. Make sure your phone and PC are on same WiFi"
echo "   if you want to access from PC browser"
echo ""
echo "2. First run will download Vosk models (~50MB each)"
echo "   This is normal and only happens once"
echo ""
echo "3. Keep Termux running in background"
echo "   Don't close it while using the app"
echo ""
echo "4. For microphone access, grant permission when asked"
echo ""
echo "========================================="
