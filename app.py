import io
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash-lite"
VOICE = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs George
MAX_TURNS = 20

st.set_page_config(page_title="My voice agent", page_icon="")

def secret(name):
    """Get a key from Streamlit Secrets, or None if missing."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


if not secret("GEMINI_API_KEY"):
    st.error("No GEMINI_API_KEY in this app's Secrets.")
    st.stop()


@st.cache_resource
def gemini():
    return genai.Client(api_key=secret("GEMINI_API_KEY"))


gem = gemini()

# Each visitor has their own state.
for key, start in {
    "todos": [],
    "history": [],
    "log": [],
    "turn": 0,
    "reply_audio": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = start

TODOS = st.session_state.todos

# Agent instructions
AGENT_NAME = "Pip"

SYSTEM_PROMPT = """You are Pip, a voice to-do assistant for a busy student.
Everything you say is read aloud, so use one or two short sentences,
no lists, no markdown, and no emojis.
Use your tools to add, read and finish to-dos.
Never say you did something unless a tool did it.
If the user asks for something none of your tools can do,
say so plainly and do nothing."""


# Tools
def add_todo(task: str) -> str:
    """Add a task to the user's to-do list."""
    TODOS.append({"task": task, "done": False})
    return f"Added: {task}. The list has {len(TODOS)} items."


def list_todos() -> str:
    """Read the user's to-do list with item numbers and status."""
    if not TODOS:
        return "The list is empty."

    return "\n".join(
        f"{i + 1}. {t['task']} "
        f"({'done' if t['done'] else 'open'})"
        for i, t in enumerate(TODOS)
    )


def mark_done(number: int) -> str:
    """Mark a to-do as done by its number, starting at 1."""
    if not 1 <= number <= len(TODOS):
        return (
            f"There is no item {number}. "
            f"The list has {len(TODOS)} items."
        )

    TODOS[number - 1]["done"] = True
    return f"Done: {TODOS[number - 1]['task']}."


TOOLS = [add_todo, list_todos, mark_done]


def hear(audio_bytes):
    """Transcribe audio using Gemini."""
    r = gem.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav",
            ),
            (
                "Transcribe this audio exactly. Output only the words. "
                "If nobody speaks, output nothing."
            ),
        ],
    )
    return (r.text or "").strip()


def think(text):
    """Send the user's message to Gemini with tools and conversation history."""
    chat = gem.chats.create(
        model=MODEL,
        history=st.session_state.history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
            max_output_tokens=300,
        ),
    )

    reply = (
        chat.send_message(text).text
        or "Sorry, I didn't catch that."
    )

    new_history = chat.get_history()
    previous_length = len(st.session_state.history)
    new_messages = new_history[previous_length:]
    st.session_state.history = new_history

    used = [ 
        f"{p.function_call.name}({dict(p.function_call.args or {})})"
        for c in new_messages
        for p in (c.parts or [])
        if p.function_call
    ]

    return reply, used


def speak(text):
    """Use ElevenLabs if configured; otherwise use free gTTS."""
    if secret("ELEVEN_API_KEY"):
        try:
            from elevenlabs.client import ElevenLabs

            eleven = ElevenLabs(
                api_key=secret("ELEVEN_API_KEY")
            )

            return b"".join(
                eleven.text_to_speech.convert(
                    voice_id=VOICE,
                    text=text,
                    model_id="eleven_flash_v2_5",
                )
            )

        except Exception:
            st.toast(
                "ElevenLabs failed. Using the free voice instead."
            )

    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text).write_to_fp(buf)
    return buf.getvalue()


# Page
st.title("🎙️ " + AGENT_NAME)
st.caption(
    "Tap the mic, speak, tap again to stop. "
    "Or type at the bottom."
)

with st.sidebar:
    st.subheader("To-do list")

    if not TODOS:
        st.caption("Empty")

    for t in TODOS:
        if t["done"]:
            st.markdown(f"✅ ~~{t['task']}~~2")
        else:
            st.markdown(f"▫️ {t['task']}")

    st.caption(
        f"Turns used: {st.session_state.turn} of {MAX_TURNS}"
    )

    if st.button("Start over"):
        for key in [
            "todos",
            "history",
            "log",
            "turn",
            "reply_audio",
        ]:
            del st.session_state[key]
        st.rerun()


# Conversation history
for who, text, used in st.session_state.log:
    with st.chat_message(
        "user" if who == "you" else "assistant"
    ):
        st.write(text)

        for u in used:
            st.caption(f"🔧 {u}")


# Play the latest response once
if st.session_state.reply_audio:
    st.audio(
        st.session_state.reply_audio,
        format="audio/mp3",
        autoplay=True,
    )
    st.session_state.reply_audio = None


# User input
audio = st.audio_input(
    "Talk to it",
    key=f"mic{st.session_state.turn}",
)
typed = st.chat_input("...or type here")

said = None

if audio:
    with st.spinner("Listening..."):
        said = hear(audio.getvalue())

    if not said:
        st.warning(
            "I didn't hear any words. Try again, "
            "a little closer to the mic."
        )

elif typed:
    said = typed


# Process the request
if said:
    if st.session_state.turn >= MAX_TURNS:
        st.warning(
            "This demo has used all its turns. "
            "Press Start over in the sidebar."
        )
        st.stop()

    with st.spinner("Thinking..."):
        reply, used = think(said)

    with st.spinner("Speaking..."):
        st.session_state.reply_audio = speak(reply)

    st.session_state.log += [
        ("you", said, []),
        ("agent", reply, used),
    ]

    st.session_state.turn += 1
    st.rerun()
