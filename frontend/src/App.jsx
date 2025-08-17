import { useState, useRef, useEffect } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesRef = useRef(null);

  // scroll to bottom when messages change
  useEffect(() => {
    const el = messagesRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length]);

  // ============ TEXT CHAT ============
  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { type: "human", content: input }];
    setMessages(newMessages);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "user1", message: input }),
      });

      const data = await res.json();
      setMessages((prev) => [
        ...newMessages,
        { type: "ai", content: data.response || "⚠️ No response" },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...newMessages,
        { type: "ai", content: "⚠️ Error connecting to backend" },
      ]);
      console.error(err);
    }

    setInput("");
  };

  // ============ VOICE CHAT ============
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("user_id", "user1");
        formData.append("file", audioBlob, "voice.wav");

        try {
          const res = await fetch("http://127.0.0.1:8000/chat_audio", {
            method: "POST",
            body: formData,
          });

          const data = await res.json();

          setMessages((prev) => [
            ...prev,
            { type: "human", content: data.original_text },
            { type: "ai", content: data.chatbot_response_local },
          ]);

          if (data.audio_file) {
            const audio = new Audio(`http://127.0.0.1:8000/${data.audio_file}`);
            audio.play();
          }
        } catch (err) {
          setMessages((prev) => [
            ...prev,
            { type: "ai", content: "⚠️ Error in voice chat" },
          ]);
          console.error(err);
        }
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      console.error("Microphone access denied", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div className="center-page">
      <div className="chat-container" role="main" aria-label="Chat container">
        <div className="chat-header">🌱 KrishiMitra Chatbot</div>

        <div className="messages" ref={messagesRef}>
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`message ${m.type === "human" ? "human" : "ai"}`}
            >
              {m.content}
            </div>
          ))}
        </div>

        <div className="input-area">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button className="btn btn-send" onClick={sendMessage}>Send</button>
        </div>

        <div style={{ padding: 12, display: "flex", justifyContent: "center", background: "#343541", borderTop: "1px solid rgba(255,255,255,0.04)" }}>
          {!recording ? (
            <button className="btn btn-record" onClick={startRecording}>🎤 Start Recording</button>
          ) : (
            <button className="btn" style={{ background: "#6b7280", color: "#fff" }} onClick={stopRecording}>⏹ Stop Recording</button>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
