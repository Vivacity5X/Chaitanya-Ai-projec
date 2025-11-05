import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_socketio import SocketIO

# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OpenRouter API key in .env file!")

app = Flask(__name__)
# A secret key is needed for session management
app.config['SECRET_KEY'] = 'your-very-secret-key' 
socketio = SocketIO(app)

# --- The AI's "Brain" and Rules ---
SYSTEM_PROMPT = {
    "role": "system", "content": (
        "You are Professor Vedika, from JSPM's RSCOE, Pune. You are on a professional call with a parent. "
        "You must be polite and follow these steps exactly. Keep all responses very short.\n"
        "1. Your first response is ONLY to greet them. Just say 'Hello'.\n"
        "2. After they respond, introduce yourself. Example: 'This is Professor Vedika calling from JSPM's RSCOE.'\n"
        "3. After they acknowledge, state the reason for the call. Example: 'I am calling to inform you that your child was absent today. May I know the reason?'\n"
        "4. After they give a reason, your final response MUST BE: 'Thank you for informing me. Take care, goodbye!'"
    )
}

# Store conversation history for each user session
conversation_history = []

def get_ai_response():
    """Sends the current conversation history to the DeepSeek AI and gets a reply."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"model": "deepseek/deepseek-chat", "messages": conversation_history}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        return reply
    except Exception as e:
        print(f"❌ API Error: {e}")
        return "I am sorry, I have a connection issue at the moment."

# --- 2. FLASK ROUTES AND SOCKETIO EVENTS ---

@app.route('/')
def index():
    """Serves the main webpage."""
    return render_template('index.html')

@socketio.on('start_call')
def handle_start_call():
    """Triggered when the user clicks 'Start Call' on the webpage."""
    global conversation_history
    print("📞 Call started from webpage.")
    # Reset conversation history for a new call
    conversation_history = [SYSTEM_PROMPT]
    
    # Get the AI's very first line (e.g., "Hello")
    ai_reply = get_ai_response()
    conversation_history.append({"role": "assistant", "content": ai_reply})
    
    # Send the reply back to the browser to be spoken
    socketio.emit('ai_response', {'text': ai_reply})

@socketio.on('user_speech')
def handle_user_speech(data):
    """Triggered when the browser recognizes the user's speech."""
    global conversation_history
    user_text = data['text'].lower()
    print(f"Parent said: {user_text}")
    
    conversation_history.append({"role": "user", "content": user_text})
    
    # Check if parent wants to end the call
    parent_end_keywords = ["bye", "ok thank you", "theek hai", "okay", "thanks"]
    if any(word in user_text for word in parent_end_keywords):
        ai_reply = "Thank you for informing me. Take care, goodbye!"
        # No need to add to history, just send final reply
        socketio.emit('ai_response', {'text': ai_reply, 'end_call': True})
        return

    # If not ending, get the next AI response
    ai_reply = get_ai_response()
    conversation_history.append({"role": "assistant", "content": ai_reply})
    
    # Check if the AI's response is the final one
    end_call = "goodbye" in ai_reply.lower()
    socketio.emit('ai_response', {'text': ai_reply, 'end_call': end_call})

# --- 3. RUN THE APPLICATION ---

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open your web browser and go to http://127.0.0.1:5000")
    # This command is simpler and more stable
    socketio.run(app, host='127.0.0.1', port=5000)