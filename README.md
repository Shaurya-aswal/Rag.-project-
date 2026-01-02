# 🎬 YouTube RAG Chatbot - Universal Multi-Language Support

A modern web application that lets you chat with YouTube videos using AI. **Works with videos in ANY language** - automatically detects available transcripts, translates them to English using Google Gemini AI, and provides intelligent responses in English.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-purple.svg)
![AI](https://img.shields.io/badge/AI-Google%20Gemini-red.svg)

## ✨ Key Features

- 🌍 **Universal Language Support**: Process videos with captions in ANY language
- 🤖 **AI Translation**: Automatic translation to English using Google Gemini
- 💬 **Smart Chat**: Ask questions about video content in natural language
- 🔍 **Vector Search**: FAISS-powered similarity search for accurate context retrieval
- 🌐 **Modern Web UI**: Beautiful, responsive interface that works on all devices
- ⚡ **High Performance**: FastAPI backend with async processing

## 🚀 Quick Start (3 Steps)

### 1. Start the Application
```bash
cd "/Users/apple/rag implmentation "
python app.py
```

### 2. Open Web Interface
Navigate to: **http://localhost:8000**

### 3. Process & Chat
1. Paste any YouTube URL (in any language)
2. Click "Process Video" 
3. Start chatting in English about the video content!

## 🎯 What Makes This Special

### Multi-Language Processing
- ✅ **Automatic Language Detection**: Finds available transcripts in any language
- ✅ **Smart Translation**: Uses Google Gemini AI for accurate translation
- ✅ **English Responses**: Always responds in clear, fluent English
- ✅ **Fallback Support**: Handles videos with auto-generated or manual captions

### Supported Video Types
- 📚 Educational content (lectures, courses)
- 🎥 Entertainment (movies, shows with captions)
- 🎤 Interviews and podcasts
- 📰 News and documentaries
- 💼 Business presentations
- 🌍 International content in any language

## 🛠️ Technical Stack

- **Backend**: FastAPI (Python)
- **AI Models**: Google Gemini 2.5 Flash
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers
- **Framework**: LangChain
- **Frontend**: Modern HTML/CSS/JavaScript
- **Translation**: Google Gemini AI

## 📋 Requirements

- Python 3.8+
- Virtual environment (already configured)
- Internet connection (for YouTube access and AI APIs)

## 🎮 Usage Examples

### Sample Questions You Can Ask:
- "What is the main topic of this video?"
- "Summarize the key points discussed"
- "What examples are given?"
- "What are the speaker's conclusions?"
- "Explain the concept mentioned at 5:30"

### Video URL Examples:
```
Educational: https://www.youtube.com/watch?v=VIDEO_ID
International: https://www.youtube.com/watch?v=VIDEO_ID (any language)
Interviews: https://www.youtube.com/watch?v=VIDEO_ID
Documentaries: https://www.youtube.com/watch?v=VIDEO_ID
```

## 🔧 API Endpoints

- **GET /**: Web interface
- **POST /process-video**: Process YouTube video with multi-language support
- **POST /chat**: Chat with processed video content
- **GET /status**: System status and video info
- **GET /health**: Health check

## 📱 Web Interface Features

- 🎨 **Modern Design**: Clean, intuitive interface
- 📱 **Mobile Responsive**: Works perfectly on phones and tablets
- ⚡ **Real-time Chat**: WhatsApp-style messaging
- 📊 **Progress Tracking**: Visual feedback during processing
- 🌐 **Cross-browser**: Compatible with all modern browsers

## 🌍 Multi-Language Support Details

### How It Works:
1. **Detection**: Automatically finds available transcripts
2. **Selection**: Chooses the best quality transcript available
3. **Translation**: Uses Google Gemini AI to translate to English
4. **Processing**: Creates vector embeddings for intelligent search
5. **Response**: Provides accurate answers in English

### Supported Languages:
- 🇬🇧 English (native support)
- 🇮🇳 Hindi, Tamil, Bengali, Telugu, Marathi
- 🇪🇸 Spanish, 🇫🇷 French, 🇩🇪 German, 🇮🇹 Italian
- 🇵🇹 Portuguese, 🇷🇺 Russian, 🇯🇵 Japanese, 🇰🇷 Korean
- 🇨🇳 Chinese (Simplified/Traditional), 🇸🇦 Arabic
- **And many more!** (Any language with YouTube captions)

## 🔍 Troubleshooting

### Common Issues:

**"No transcripts available"**
- The video doesn't have captions/subtitles
- Try videos from educational channels or with closed captions

**"Processing taking long"**
- Translation takes time for longer videos
- First run downloads AI models (~90MB)

**"Server not starting"**
- Port 8000 might be in use
- Check if dependencies are installed

### System Requirements:
- **RAM**: 2GB+ recommended
- **Storage**: 1GB+ for AI models
- **Internet**: Required for YouTube and AI APIs

## 📂 Project Structure

```
rag implmentation/
├── app.py                 # Main FastAPI server with multi-language support
├── index.html            # Modern web interface  
├── demo.py              # Demo and testing script
├── rag.ipynb            # Original research notebook
├── requirements.txt     # Python dependencies
├── README.md           # This documentation
└── .venv/              # Virtual environment
```

## 🧪 Testing

Run the demo script to test functionality:
```bash
python demo.py
```

This will:
- Test server health
- Process a sample video
- Demonstrate chat functionality
- Show system capabilities

## 🤝 Contributing

This project demonstrates advanced RAG (Retrieval-Augmented Generation) with:
- Multi-language transcript processing
- AI-powered translation
- Vector similarity search
- Modern web interface design
- Production-ready FastAPI architecture

## 🎉 Success Stories

**Before**: Limited to English-only YouTube videos
**After**: Universal support for videos in ANY language with intelligent English responses

## 🔒 Privacy & Security

- **Local Processing**: All AI processing uses your API keys
- **No Data Storage**: Chat history is session-based
- **Secure APIs**: Uses official Google AI services

---

## 🚀 **Ready to Chat with Any YouTube Video?**

**Start the server and begin exploring global content in any language!**

```bash
python app.py
```

**Then open: http://localhost:8000**

**🌍 The world of YouTube content is now accessible in English! 🤖💬**
# Rag.-project-
