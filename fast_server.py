#!/usr/bin/env python3
"""
Multilingual YouTube RAG Chatbot - Supports Any Language
"""

import os
import logging
import warnings
import json

# Disable threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Google API Key from environment variable or use default
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDbfYrhI_NnRl-CscXQ0m4dQ6Z0weM-EPk")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
    logger.error("❌ Please set your Google API Key!")
    logger.info("💡 Set it as environment variable: export GOOGLE_API_KEY='your_actual_key'")
    logger.info("💡 Or update the GOOGLE_API_KEY variable in fast_server.py")

# Simple in-memory storage
app_state = {
    "video_id": None,
    "original_transcript": "",
    "english_transcript": "",
    "original_language": "en",
    "chunks": [],
    "is_ready": False
}

def detect_language(text):
    """Detect language of text"""
    try:
        from langdetect import detect
        detected = detect(text[:1000])
        logger.info(f"Detected language: {detected}")
        return detected
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"

def translate_to_english(text, source_lang):
    """Translate text to English if needed"""
    if source_lang == "en":
        return text
    
    # Try multiple translation methods
    translation_methods = [
        ('deep_translator.GoogleTranslator', 'Google Translate'),
        ('googletrans.Translator', 'Alternative Google Translate'),
    ]
    
    for method_path, method_name in translation_methods:
        try:
            logger.info(f"Trying translation method: {method_name}")
            
            if method_path == 'deep_translator.GoogleTranslator':
                from deep_translator import GoogleTranslator
                
                # Split into smaller chunks for translation
                chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
                translated_chunks = []
                
                translator = GoogleTranslator(source=source_lang, target='en')
                
                logger.info(f"Translating {len(chunks)} chunks from {source_lang} to English using {method_name}")
                
                for i, chunk in enumerate(chunks):
                    try:
                        translated = translator.translate(chunk)
                        translated_chunks.append(translated)
                        logger.info(f"✅ Translated chunk {i+1}/{len(chunks)}")
                    except Exception as e:
                        logger.warning(f"❌ Translation failed for chunk {i+1}: {e}")
                        # Keep original text for failed chunks
                        translated_chunks.append(chunk)
                
                result = " ".join(translated_chunks)
                if any("translated" in str(chunk) for chunk in translated_chunks):
                    logger.info(f"✅ Translation completed using {method_name}")
                    return result
                    
            elif method_path == 'googletrans.Translator':
                try:
                    import googletrans
                    translator = googletrans.Translator()
                    
                    # Split into smaller chunks
                    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
                    translated_chunks = []
                    
                    logger.info(f"Translating {len(chunks)} chunks using {method_name}")
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            result = translator.translate(chunk, src=source_lang, dest='en')
                            translated_chunks.append(result.text)
                            logger.info(f"✅ Translated chunk {i+1}/{len(chunks)}")
                        except Exception as e:
                            logger.warning(f"❌ Translation failed for chunk {i+1}: {e}")
                            translated_chunks.append(chunk)
                    
                    final_result = " ".join(translated_chunks)
                    logger.info(f"✅ Translation completed using {method_name}")
                    return final_result
                    
                except ImportError:
                    logger.warning(f"{method_name} not available, installing...")
                    import subprocess
                    subprocess.run(['pip', 'install', 'googletrans==4.0.0rc1'], check=False)
                    
        except Exception as e:
            logger.warning(f"Translation method {method_name} failed: {e}")
            continue
    
    # If all translation methods fail, return original text with a note
    logger.warning("All translation methods failed, returning original text")
    return f"[Original {source_lang.upper()} text - Translation failed]: {text}"

def extract_video_id(url_or_id):
    """Extract video ID from URL"""
    if "youtube.com" in str(url_or_id):
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in str(url_or_id):
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return str(url_or_id)

def simple_chunk_text(text, chunk_size=800):
    """Simple text chunking without complex dependencies"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def simple_search(query, chunks, top_k=3):
    """Enhanced search for multilingual content"""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    # If no chunks or empty query, return first few chunks
    if not chunks or not query.strip():
        return chunks[:top_k] if chunks else []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        
        # Direct keyword matching
        for word in query_words:
            if word in chunk_lower:
                score += 10
        
        # Fuzzy matching for common terms
        common_terms = {
            'langchain': ['langchain', 'लैंग चेन', 'लांगचेन'],
            'rag': ['rag', 'रैग', 'retrieval'],
            'ai': ['ai', 'artificial', 'intelligence', 'एआई'],
            'machine learning': ['machine', 'learning', 'मशीन', 'लर्निंग'],
            'python': ['python', 'पायथन'],
            'tutorial': ['tutorial', 'ट्यूटोरियल', 'सिखाना', 'learn'],
            'video': ['video', 'वीडियो', 'channel', 'चैनल'],
            'model': ['model', 'मॉडल'],
            'data': ['data', 'डेटा'],
            'code': ['code', 'कोड', 'coding', 'कोडिंग']
        }
        
        # Check for fuzzy matches
        for term, variations in common_terms.items():
            if any(word in term.lower() for word in query_words):
                for variation in variations:
                    if variation in chunk_lower:
                        score += 5
        
        # Length-based bonus for substantial chunks
        if len(chunk) > 200:
            score += 2
        
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # If no scored chunks found, return first few chunks for general questions
    if not scored_chunks:
        logger.info(f"No keyword matches found for '{query}', returning first chunks")
        return chunks[:top_k]
    
    scored_chunks.sort(reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k]]

def fetch_transcript_smart(video_id):
    """
    Smart transcript fetching with comprehensive language support
    """
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
    
    logger.info(f"Attempting to fetch transcript for video: {video_id}")
    
    try:
        # Create API instance and get all available transcripts
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Log available transcripts for debugging
        logger.info("Available transcripts:")
        available_languages = []
        
        for transcript in transcript_list:
            lang_info = f"{transcript.language_code} ({transcript.language})"
            if transcript.is_generated:
                lang_info += " [auto-generated]"
            if transcript.is_translatable:
                lang_info += " [translatable]"
            logger.info(f"  - {lang_info}")
            available_languages.append(transcript.language_code)
        
        # Priority order: English first, then any available language
        preferred_languages = ['en', 'en-US', 'en-GB'] + [lang for lang in available_languages if lang not in ['en', 'en-US', 'en-GB']]
        
        transcript_data = None
        original_lang = None
        
        # Try to get transcript in preferred order
        for lang_code in preferred_languages:
            try:
                logger.info(f"Trying to fetch transcript in: {lang_code}")
                
                # Try direct fetch if language is available
                if lang_code in available_languages:
                    transcript = transcript_list.find_transcript([lang_code])
                    transcript_data = transcript.fetch()
                    original_lang = lang_code
                    logger.info(f"✅ Successfully fetched transcript in: {lang_code}")
                    break
                    
            except Exception as e:
                logger.info(f"❌ Failed to fetch {lang_code}: {e}")
                continue
        
        # If no transcript found, try first available
        if not transcript_data:
            logger.info("No preferred language found, trying first available transcript...")
            try:
                first_transcript = next(iter(transcript_list))
                transcript_data = first_transcript.fetch()
                original_lang = first_transcript.language_code
                logger.info(f"✅ Fetched first available transcript: {original_lang}")
                
                # Check if YouTube can translate it to English
                if first_transcript.is_translatable and original_lang not in ['en', 'en-US', 'en-GB']:
                    try:
                        logger.info("Attempting to get YouTube English translation...")
                        english_transcript_obj = first_transcript.translate('en')
                        english_transcript_data = english_transcript_obj.fetch()
                        
                        # Return both original and English versions
                        original_text = " ".join([item.text for item in transcript_data])
                        english_text = " ".join([item.text for item in english_transcript_data])
                        
                        logger.info(f"✅ Got YouTube translation: {original_lang} -> en")
                        return original_text, english_text, original_lang, 'en'
                        
                    except Exception as e:
                        logger.warning(f"YouTube translation failed: {e}")
                
            except Exception as e:
                logger.error(f"Failed to get any transcript: {e}")
                raise Exception(f"No transcripts available for video {video_id}")
        
        if not transcript_data:
            raise Exception(f"Could not fetch any transcript for video {video_id}")
        
        # Convert to text using correct attribute access
        transcript_text = " ".join([item.text for item in transcript_data])
        
        # If we got English directly, return it
        if original_lang in ['en', 'en-US', 'en-GB']:
            logger.info("Transcript is already in English")
            return transcript_text, transcript_text, original_lang, original_lang
        
        # Otherwise, we'll need to translate using our translation service
        logger.info(f"Need to translate {original_lang} to English using external service")
        return transcript_text, None, original_lang, None
        
    except TranscriptsDisabled:
        raise Exception(f"Transcripts are disabled for video {video_id}")
    except NoTranscriptFound:
        raise Exception(f"No transcripts found for video {video_id}")
    except Exception as e:
        logger.error(f"Unexpected error fetching transcript: {e}")
        raise Exception(f"Failed to fetch transcript: {str(e)}")

def setup_video_simple(video_id):
    """Fast video setup with multilingual support"""
    global app_state
    
    try:
        video_id = extract_video_id(video_id)
        logger.info(f"Processing multilingual video: {video_id}")
        
        # Use smart transcript fetching
        try:
            # Get transcript using smart fetch
            transcript_result = fetch_transcript_smart(video_id)
            
            if len(transcript_result) == 4:
                # We got both original and English versions
                original_transcript, english_transcript, original_lang, english_lang = transcript_result
                logger.info(f"Got pre-translated transcript: {original_lang} -> {english_lang}")
            else:
                # We got original only, need to translate
                original_transcript, _, original_lang, _ = transcript_result
                english_transcript = None
                
            logger.info(f"Original transcript length: {len(original_transcript)} chars, language: {original_lang}")
            
            # Translate to English if needed and not already provided
            if not english_transcript:
                if original_lang not in ['en', 'en-US', 'en-GB']:
                    logger.info(f"Translating from {original_lang} to English...")
                    english_transcript = translate_to_english(original_transcript, original_lang)
                else:
                    english_transcript = original_transcript
                    logger.info("Transcript is already in English")
            else:
                logger.info("Using YouTube-provided English translation")
                
        except Exception as e:
            logger.error(f"Smart transcript fetch failed: {e}")
            raise Exception(f"Could not fetch transcript: {str(e)}")
        
        # Simple chunking
        chunks = simple_chunk_text(english_transcript)
        
        app_state.update({
            "video_id": video_id,
            "original_transcript": original_transcript,
            "english_transcript": english_transcript,
            "original_language": original_lang,
            "chunks": chunks,
            "is_ready": True
        })
        
        success_msg = f"Video {video_id} processed! Language: {original_lang}, Chunks: {len(chunks)}"
        logger.info(success_msg)
        return True, success_msg
        
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return False, str(e)

def chat_simple(question):
    """Fast chat with multilingual support"""
    if not app_state["is_ready"]:
        return "Please set up a video first!"
    
    try:
        # Find relevant chunks in English (or translated content)
        relevant_chunks = simple_search(question, app_state["chunks"])
        context = "\n\n".join(relevant_chunks)
        
        # Get language info
        original_lang = app_state["original_language"]
        lang_note = f" (originally in {original_lang.upper()})" if original_lang != "en" else ""
        
        # Check if we have meaningful content
        if not context.strip():
            return f"No relevant content found for your question in the video{lang_note}. Try rephrasing your question or ask about general video content."
        
        # Try Google GenAI directly first, then fallback to LangChain
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Enhanced prompt for multilingual content - let AI handle translation if needed
            if original_lang != "en":
                prompt = f"""You are analyzing content from a YouTube video that was originally in {original_lang.upper()}. The content below may be in {original_lang.upper()} or partially translated to English.

Video Context:
{context[:4000]}

User Question: {question}

Instructions:
- If the content is in {original_lang.upper()}, translate it to English and provide your analysis
- Answer in clear, fluent English
- Provide a comprehensive and helpful response
- Focus on the main topics and key information from the video
- If you can understand the original language, use that understanding to provide better context

Answer in English:"""
            else:
                prompt = f"""Based on the following video transcript, provide a helpful and comprehensive answer to the user's question in clear English.

Video Context:
{context[:4000]}

User Question: {question}

Instructions:
- Answer in clear, fluent English
- Be comprehensive and helpful
- Focus on the main topics and key information from the video

Answer:"""
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e1:
            logger.warning(f"Direct GenAI failed: {e1}")
            
            # Check if it's a quota issue
            if "quota" in str(e1).lower() or "429" in str(e1):
                quota_msg = f"""🚫 **AI Quota Exceeded**: The Google Gemini API has reached its daily limit (20 requests on free tier).

📋 **Based on the video content{lang_note}:**

{context[:1200] if context else 'No relevant content found for your question.'}

💡 **Your question**: "{question}"

🔄 **What you can do:**
- Try again in a few hours when the quota resets
- The content above should help answer your question
- Ask more specific questions to find relevant sections
- The system successfully processed the video and found relevant content"""
                
                return quota_msg
            
            # Try LangChain fallback
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.1
                )
                
                response = llm.invoke(prompt)
                return response.content
                
            except Exception as e2:
                logger.error(f"LangChain GenAI also failed: {e2}")
                
                # Enhanced fallback response for multilingual content
                if original_lang != "en":
                    fallback_msg = f"""🌍 **Multilingual Content Analysis{lang_note}**

📋 **Relevant video content for your question: "{question}"**

{context[:1000] if context else 'No specific content found for your question.'}

ℹ️ **Note**: This video was originally in {original_lang.upper()}. The AI response system is currently unavailable, but the content above relates to your question. 

💡 **Tips**: 
- Try rephrasing your question for better search results
- Ask about specific topics mentioned in the video
- The system successfully extracted and processed the video content"""
                else:
                    fallback_msg = f"""📋 **Video Content Analysis**

**Relevant content for your question: "{question}"**

{context[:1000] if context else 'No specific content found for your question.'}

ℹ️ **Note**: The AI response system is currently experiencing issues, but the content above relates to your question.

💡 **Tip**: Try rephrasing your question or asking about specific topics for better results."""
                
                return fallback_msg
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"Sorry, error processing your question: {e}"

@app.route('/')
def index():
    try:
        return render_template_string(open('index.html').read())
    except FileNotFoundError:
        return {
            "status": "YouTube AI Chat Assistant API",
            "version": "1.0.0",
            "description": "Multilingual YouTube RAG Chatbot API Server",
            "endpoints": {
                "/api/status": "GET - Check server status",
                "/api/setup": "POST - Setup video for chat (requires video_id)",
                "/api/chat": "POST - Chat with video (requires question)"
            },
            "features": [
                "Supports videos in ANY language",
                "Auto-translates transcripts to English", 
                "AI-powered responses about video content"
            ],
            "chrome_extension": "Use with YouTube AI Chat Assistant Chrome Extension"
        }

@app.route('/api')
def api_info():
    return {
        "status": "YouTube AI Chat Assistant API",
        "version": "1.0.0",
        "description": "Multilingual YouTube RAG Chatbot API Server",
        "endpoints": {
            "/api/status": "GET - Check server status",
            "/api/setup": "POST - Setup video for chat (requires video_id)",
            "/api/chat": "POST - Chat with video (requires question)"
        },
        "features": [
            "Supports videos in ANY language",
            "Auto-translates transcripts to English", 
            "AI-powered responses about video content"
        ],
        "chrome_extension": "Use with YouTube AI Chat Assistant Chrome Extension"
    }

@app.route('/api/setup', methods=['POST'])
def setup_endpoint():
    try:
        data = request.get_json()
        video_id = data.get('video_id', '').strip()
        
        if not video_id:
            return jsonify({'error': 'Video ID required'}), 400
        
        success, message = setup_video_simple(video_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'video_id': app_state['video_id'],
                'original_language': app_state['original_language'],
                'chunks_count': len(app_state['chunks'])
            })
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question required'}), 400
        
        response = chat_simple(question)
        
        return jsonify({
            'response': response,
            'video_id': app_state['video_id'],
            'original_language': app_state['original_language']
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/api/status')
def status_endpoint():
    return jsonify({
        'ready': app_state['is_ready'],
        'video_id': app_state['video_id'],
        'original_language': app_state.get('original_language', 'en'),
        'chunks_count': len(app_state['chunks']),
        'has_translation': app_state.get('original_language', 'en') != 'en'
    })

if __name__ == '__main__':
    print("🌍 Multilingual YouTube RAG Chatbot")
    print("📍 Local: http://localhost:8000")
    print("📍 Network: http://0.0.0.0:8000")
    print("🔤 Supports videos in ANY language!")
    print("🔄 Auto-translates to English")
    print("💬 Responds in English")
    print("=" * 50)
    
    app.run(
        host='0.0.0.0',  # Bind to all network interfaces
        port=8000, 
        debug=False, 
        threaded=True,
        use_reloader=False
    )
