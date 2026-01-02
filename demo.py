#!/usr/bin/env python3
"""
Demo script for YouTube RAG Chatbot
Shows various features and capabilities
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def demo_print(message, style="info"):
    """Pretty print demo messages"""
    styles = {
        "info": "💡",
        "success": "✅", 
        "processing": "⚙️",
        "chat": "💬",
        "error": "❌"
    }
    print(f"{styles.get(style, '•')} {message}")

def main():
    """Run demo scenarios"""
    print("🎬 YouTube RAG Chatbot Demo")
    print("=" * 50)
    
    # Check health
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            demo_print("Server is running and healthy!", "success")
        else:
            demo_print("Server health check failed!", "error")
            return
    except:
        demo_print("Server is not running! Please start it with: python app.py", "error")
        return
    
    print("\n📺 DEMO SCENARIO 1: Educational Video")
    print("-" * 40)
    
    # Demo 1: Working video with English transcript
    educational_video = "https://www.youtube.com/watch?v=TBweeDDJeTk"  # Known working video
    demo_print(f"Processing educational video: {educational_video}", "processing")
    
    try:
        response = requests.post(f"{API_BASE}/process-video", 
                               json={"youtube_url": educational_video})
        if response.status_code == 200:
            data = response.json()
            demo_print("Video processed successfully!", "success")
            demo_print(f"Video ID: {data['video_id']}")
            demo_print(f"Preview: {data['transcript_preview'][:100]}...")
        else:
            demo_print(f"Processing failed: {response.json()['detail']}", "error")
            return
    except Exception as e:
        demo_print(f"Error: {e}", "error")
        return
    
    # Give it a moment to fully process
    time.sleep(2)
    
    # Demo questions for educational video
    questions = [
        "What is the main topic of this video?",
        "Can you summarize the key points discussed?",
        "What are the main takeaways?",
        "Are there any specific examples mentioned?"
    ]
    
    for question in questions:
        demo_print(f"Asking: {question}", "chat")
        try:
            response = requests.post(f"{API_BASE}/chat", json={"question": question})
            if response.status_code == 200:
                answer = response.json()["answer"]
                print(f"   🤖 Answer: {answer[:150]}...")
                print()
            else:
                demo_print(f"Chat failed: {response.json()}", "error")
        except Exception as e:
            demo_print(f"Error: {e}", "error")
        
        time.sleep(1)  # Be nice to the API
    
    print("\n📊 System Status Check")
    print("-" * 40)
    
    try:
        response = requests.get(f"{API_BASE}/status")
        status = response.json()
        demo_print(f"System Ready: {status['is_ready']}", "success")
        demo_print(f"Current Video: {status['video_id']}")
        demo_print(f"RAG Chain Active: {status['has_rag_chain']}")
    except:
        demo_print("Status check failed", "error")
    
    print("\n" + "=" * 50)
    demo_print("Demo completed! The web interface is available at:", "success")
    print("   🌐 http://localhost:8000")
    print()
    demo_print("Try these features in the web app:", "info")
    print("   • Process different YouTube videos")
    print("   • Ask various types of questions")
    print("   • Real-time chat interface")
    print("   • Mobile-friendly responsive design")
    print()
    demo_print("Sample YouTube videos you can try:", "info")
    print("   • Educational: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print("   • Tech talks: Any conference presentation")
    print("   • Tutorials: Any how-to video")
    print("   • Interviews: Podcast episodes")

if __name__ == "__main__":
    main()
