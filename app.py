import json
import os
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import requests

BASE_DIR = Path(__file__).parent
CHAT_DIR = BASE_DIR / "chats"
MEMORY_PATH = BASE_DIR / "memory.json"
LOG_PATH = BASE_DIR / "ai_interaction_log.md"
HF_MODEL = "deepseek-ai/DeepSeek-R1:fastest"
STREAM_DELAY_SEC = 0.03

DEFAULT_MEMORY = {
    "hiking": "true",
    "name": "User",
    "preferred_language": "English",
    "interests": ["outdoors", "history", "food"],
    "communication_style": "informative",
    "favorite_topics": ["California history", "food", "plants"],
}

SAMPLE_ASSISTANT_TEXT = (
    "Hiking is a wonderful outdoor activity that offers a great way to connect with "
    "nature and get some exercise. There are many benefits to hiking, including:\n\n"
    "1. Improves physical health: Hiking can help improve cardiovascular health, "
    "increase flexibility, and strengthen muscles.\n"
    "2. Reduces stress: Being in nature has been shown to reduce stress levels and "
    "improve mood.\n"
    "3. Connects with nature: Hiking allows you to connect with the natural world "
    "and appreciate its beauty.\n"
    "4. Promotes mindfulness: Hiking requires focus and attention, which can help "
    "promote mindfulness and self-awareness.\n"
    "5. Develops physical skills: Hiking requires skills such as navigation, "
    "route-finding, and problem-solving, which can help develop physical abilities.\n\n"
    "If you're new to hiking, it's a good idea to start with shorter, easier trails "
    "and gradually work your way up to more challenging trails. It's also a good idea to:\n\n"
    "1. Research the trail: Look up the trail and read reviews to get an idea of what to expect.\n"
    "2. Check the weather: Check the weather forecast and be prepared for changing conditions.\n"
    "3. Bring necessary gear: Bring plenty of water, snacks, and sunscreen.\n"
    "4. Let someone know your plans: Let a friend or family member know where you're going "
    "and when you plan to return.\n"
    "5. Be prepared for emergencies: Make sure you have a plan in place in case of an emergency.\n\n"
    "Some popular types of hiking include:\n\n"
    "1. Backpacking: This type of hiking involves carrying a backpack with all your gear "
    "and spending multiple days or even weeks on the trail.\n"
    "2. Day hiking: This type of hiking involves hiking for shorter periods of time, "
    "usually a day or two.\n"
    "3. Ultralight hiking: This type of hiking involves carrying minimal gear to reduce "
    "weight and increase efficiency.\n"
    "4. Backcountry hiking: This type of hiking involves hiking in remote areas with "
    "limited access to amenities.\n\n"
    "What kind of hiking do you enjoy most? Do you have a favorite trail or location?"
)


def ensure_storage():
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists() or MEMORY_PATH.read_text().strip() == "":
        MEMORY_PATH.write_text(json.dumps(DEFAULT_MEMORY, indent=2))
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# AI Interaction Log\n")


def load_memory():
    try:
        return json.loads(MEMORY_PATH.read_text())
    except Exception:
        return DEFAULT_MEMORY.copy()


def save_memory(memory):
    MEMORY_PATH.write_text(json.dumps(memory, indent=2))


def load_chat(chat_id):
    path = CHAT_DIR / f"{chat_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_chat(chat):
    chat["updated_at"] = datetime.utcnow().isoformat()
    path = CHAT_DIR / f"{chat['id']}.json"
    path.write_text(json.dumps(chat, indent=2))


def list_chats():
    chats = []
    for path in CHAT_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            chats.append(data)
        except Exception:
            continue
    chats.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return chats


def create_new_chat(title="New Chat", messages=None):
    chat_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    now = datetime.utcnow().isoformat()
    chat = {
        "id": chat_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": messages or [],
    }
    save_chat(chat)
    return chat


def format_chat_time(ts):
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return ""
    now = datetime.utcnow()
    if now - dt < timedelta(minutes=5):
        return "Now"
    return dt.strftime("%b %d")


def log_interaction(user_text, assistant_text):
    timestamp = datetime.utcnow().isoformat()
    entry = (
        f"\n## {timestamp}\n"
        f"**User:** {user_text}\n\n"
        f"**Assistant:** {assistant_text}\n"
    )
    with LOG_PATH.open("a") as f:
        f.write(entry)


def update_memory_from_text(memory, text):
    lowered = text.lower().strip()
    if lowered.startswith("i like "):
        interest = lowered.replace("i like ", "").strip(" .!")
        if interest and interest not in memory.get("interests", []):
            memory.setdefault("interests", []).append(interest)
    if "my name is" in lowered:
        name = lowered.split("my name is", 1)[1].strip(" .!")
        if name:
            memory["name"] = name.title()
    return memory


def generate_assistant_response(user_text):
    if "hike" in user_text.lower():
        return SAMPLE_ASSISTANT_TEXT
    response = (
        "Thanks for sharing! I can help with that. "
        "Tell me a bit more about what you want to explore."
    )
    return response


def force_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def get_hf_token():
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return ""
    if not token or not str(token).strip():
        return ""
    return str(token).strip()


def call_hf_test_message(token, message="Hello!"):
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
    except requests.RequestException as exc:
        return None, f"Network error while contacting Hugging Face: {exc}"

    if resp.status_code != 200:
        error_message = f"Hugging Face API error ({resp.status_code})."
        try:
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("error"):
                error_message = payload["error"]
        except ValueError:
            pass
        return None, error_message

    try:
        data = resp.json()
    except ValueError:
        return None, "Hugging Face response could not be decoded."

    if isinstance(data, dict):
        if "choices" in data and data["choices"]:
            first = data["choices"][0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict) and message.get("content"):
                    return message["content"], None
        if "error" in data:
            return None, data["error"]
    return None, "Unexpected response format from Hugging Face."


def build_chat_messages(messages, memory):
    chat = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the conversation history and user memory for context. "
                "Do not reveal internal reasoning or chain-of-thought. Respond with the final answer only."
            ),
        }
    ]
    if memory:
        chat.append({"role": "system", "content": f"User memory: {json.dumps(memory)}"})
    for msg in messages:
        chat.append({"role": msg["role"], "content": msg["content"]})
    return chat


def call_hf_with_history(token, messages, memory):
    chat_messages = build_chat_messages(messages, memory)
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"model": HF_MODEL, "messages": chat_messages}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        return None, f"Network error while contacting Hugging Face: {exc}"
    if resp.status_code != 200:
        error_message = f"Hugging Face API error ({resp.status_code})."
        try:
            payload_json = resp.json()
            if isinstance(payload_json, dict) and payload_json.get("error"):
                error_message = payload_json["error"]
        except ValueError:
            pass
        return None, error_message
    try:
        data = resp.json()
    except ValueError:
        return None, "Hugging Face response could not be decoded."
    if isinstance(data, dict) and data.get("choices"):
        first = data["choices"][0]
        message = first.get("message", {})
        if isinstance(message, dict) and message.get("content"):
            return message["content"], None
    return None, "Unexpected response format from Hugging Face."


def call_hf_stream(token, prompt):
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"model": HF_MODEL, "messages": prompt, "stream": True}
    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    except requests.RequestException as exc:
        return None, f"Network error while contacting Hugging Face: {exc}"

    if resp.status_code != 200:
        error_message = f"Hugging Face API error ({resp.status_code})."
        try:
            payload_json = resp.json()
            if isinstance(payload_json, dict) and payload_json.get("error"):
                error_message = payload_json["error"]
        except ValueError:
            pass
        return None, error_message

    return resp, None


def parse_sse_lines(response):
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line.replace("data:", "", 1).strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        yield payload


def extract_text_from_stream_payload(payload):
    if isinstance(payload, dict):
        if "choices" in payload and payload["choices"]:
            choice = payload["choices"][0]
            if isinstance(choice, dict):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    return delta.get("content", "") or ""
    return ""


def strip_think_blocks(text):
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def extract_memory_update(token, user_message):
    prompt = (
        "Extract any personal facts or preferences from the user message. "
        "Return ONLY a JSON object with keys like name, interests, preferred_language, "
        "communication_style, favorite_topics. If none, return {}.\n\n"
        f"User message: {user_message}"
    )
    response_text, error = call_hf_test_message(token, prompt)
    if error or not response_text:
        return {}
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            return data
    except ValueError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(response_text[start : end + 1])
                if isinstance(data, dict):
                    return data
            except ValueError:
                pass
    return {}


def merge_memory(existing, updates):
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, list):
            existing_list = existing.get(key, [])
            if not isinstance(existing_list, list):
                existing_list = []
            merged = list(existing_list)
            for item in value:
                if item not in merged:
                    merged.append(item)
            existing[key] = merged
        elif isinstance(value, dict):
            existing_sub = existing.get(key, {})
            if not isinstance(existing_sub, dict):
                existing_sub = {}
            existing_sub.update(value)
            existing[key] = existing_sub
        else:
            existing[key] = value
    return existing


ensure_storage()

st.set_page_config(page_title="My AI Chat", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; }
      body { background: #0f172a; }
      section[data-testid="stAppViewContainer"] { background: #0f172a; }
      section[data-testid="stSidebar"] { background: #1e293b; }
      section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p { color: #f8fafc; }
      .chat-title { font-size: 2.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 1rem; }
      .chat-shell ol { padding-left: 1.3rem; }
      .chat-shell li { margin-bottom: 0.4rem; }
      .stChatInput { margin-top: 1rem; }
      .stButton button { border-radius: 10px; }
      .clear-btn button { background: #ef4444 !important; color: white !important; border: none; }
      .recent-title { color: #cbd5f5; font-size: 0.9rem; margin: 1.5rem 0 0.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

memory = load_memory()

hf_token = get_hf_token()
if not hf_token:
    st.error(
        "Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml to enable chat."
    )
    st.sidebar.warning("HF token not loaded")
else:
    st.sidebar.success("HF token loaded")

if "current_chat_id" not in st.session_state:
    chats = list_chats()
    if not chats:
        sample_messages = [
            {"role": "user", "content": "I like hiking"},
            {"role": "assistant", "content": SAMPLE_ASSISTANT_TEXT},
        ]
        chat = create_new_chat("Hiking Adventures...", sample_messages)
        st.session_state.current_chat_id = chat["id"]
    else:
        st.session_state.current_chat_id = chats[0]["id"]

# Sidebar
st.sidebar.markdown("## Chats")
if st.sidebar.button("New Chat"):
    new_chat = create_new_chat()
    st.session_state.current_chat_id = new_chat["id"]
    st.session_state.messages = []

with st.sidebar.expander("User Memory", expanded=True):
    clear_col = st.container()
    with clear_col:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("Clear Memory", key="clear_memory"):
            memory = DEFAULT_MEMORY.copy()
            save_memory(memory)
        st.markdown("</div>", unsafe_allow_html=True)
    st.json(memory)

st.sidebar.markdown('<div class="recent-title">Recent Chats</div>', unsafe_allow_html=True)

chats = list_chats()
chat_list_container = st.sidebar.container(height=260)
with chat_list_container:
    if not chats:
        st.info("No chats yet. Click New Chat to start.")
    for chat in chats:
        title = chat.get("title", "Untitled")
        chat_id = chat.get("id")
        timestamp = format_chat_time(chat.get("updated_at", ""))
        cols = st.columns([8, 2, 1])
        is_active = st.session_state.current_chat_id == chat_id
        label = f"▶ {title}" if is_active else title
        if cols[0].button(label, key=f"select_{chat_id}", type="primary" if is_active else "secondary"):
            st.session_state.current_chat_id = chat_id
        cols[1].markdown(
            f"<div style='color:#94a3b8; font-size:0.75rem; margin-top:6px'>{timestamp}</div>",
            unsafe_allow_html=True,
        )
        if cols[2].button("✕", key=f"del_{chat_id}"):
            try:
                os.remove(CHAT_DIR / f"{chat_id}.json")
            except OSError:
                pass
            remaining = list_chats()
            if remaining:
                st.session_state.current_chat_id = remaining[0]["id"]
            else:
                st.session_state.current_chat_id = None
                st.session_state.messages = []
            force_rerun()

# Main area
st.markdown('<div class="chat-title">My AI Chat</div>', unsafe_allow_html=True)

current_chat = load_chat(st.session_state.current_chat_id) if st.session_state.current_chat_id else None
if not current_chat:
    current_chat = create_new_chat()
    st.session_state.current_chat_id = current_chat["id"]

if st.session_state.get("active_chat_id") != current_chat["id"]:
    st.session_state.active_chat_id = current_chat["id"]
    st.session_state.messages = list(current_chat.get("messages", []))

messages = st.session_state.get("messages", [])

with st.sidebar.expander("Debug Memory + Prompt", expanded=False):
    show_debug = st.checkbox("Show debug details", value=False, key="show_debug")
    if show_debug:
        st.markdown("**Current Memory**")
        st.json(memory)
        st.markdown("**Prompt Preview**")
        st.code(json.dumps(build_chat_messages(messages, memory), indent=2))

history_container = st.container(height=520, border=True)
with history_container:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

user_input = st.chat_input("Type a message and press Enter")
if user_input:
    new_msg = {"role": "user", "content": user_input}
    messages.append(new_msg)
    with history_container:
        with st.chat_message("user"):
            st.write(user_input)
    if current_chat.get("title", "New Chat") == "New Chat":
        current_chat["title"] = (user_input[:20] + "...") if len(user_input) > 20 else user_input

    if not hf_token:
        memory = update_memory_from_text(memory, user_input)
        save_memory(memory)

    if hf_token:
        chat_messages = build_chat_messages(messages, memory)
        response, error = call_hf_stream(hf_token, chat_messages)
        if error:
            assistant_text = f"Sorry, I ran into an API error: {error}"
        else:
            assistant_text = ""
            raw_text = ""
            with history_container:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    for payload in parse_sse_lines(response):
                        chunk = extract_text_from_stream_payload(payload)
                        if chunk:
                            raw_text += chunk
                            assistant_text = strip_think_blocks(raw_text)
                            placeholder.write(assistant_text)
                            time.sleep(STREAM_DELAY_SEC)
            if not assistant_text:
                assistant_text = "Sorry, I couldn't generate a response."
    else:
        assistant_text = generate_assistant_response(user_input)
        with history_container:
            with st.chat_message("assistant"):
                st.write(strip_think_blocks(assistant_text))

    assistant_text = strip_think_blocks(assistant_text)
    messages.append({"role": "assistant", "content": assistant_text})
    current_chat["messages"] = messages
    save_chat(current_chat)
    log_interaction(user_input, assistant_text)
    if hf_token:
        updates = extract_memory_update(hf_token, user_input)
        if updates:
            memory = merge_memory(memory, updates)
            save_memory(memory)
    force_rerun()
