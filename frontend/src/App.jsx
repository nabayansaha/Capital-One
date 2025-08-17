import { useState, useRef, useEffect } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesRef = useRef(null);
  const [imageFile, setImageFile] = useState(null);

  useEffect(() => {
    const el = messagesRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length]);

  // ================= SEND MESSAGE (TEXT / IMAGE / AUDIO) =================
  const sendMessage = async (audioBlob = null) => {
    if (!input.trim() && !imageFile && !audioBlob) return;

    let newMessages = [...messages];
    if (input.trim()) newMessages.push({ type: "human", content: input });
    if (imageFile) newMessages.push({ type: "human", content: `📷 Sent an image: ${imageFile.name}` });
    if (audioBlob) newMessages.push({ type: "human", content: "🎤 Sent a voice message" });
    setMessages(newMessages);

    const formData = new FormData();
    formData.append("user_id", "user1");
    if (input.trim()) formData.append("message", input);
    if (imageFile) formData.append("file", imageFile);
    if (audioBlob) formData.append("file", audioBlob, "voice.wav");

    try {
      const res = await fetch("http://127.0.0.1:8000/chat_dynamic", {
        method: "POST",
        body: formData,
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
    setImageFile(null);
  };

  // ================= IMAGE SELECT =================
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) setImageFile(e.target.files[0]);
  };

  // ================= VOICE CHAT =================
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
        await sendMessage(audioBlob); // send audio to same endpoint
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
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <input type="file" accept="image/*" onChange={handleImageChange} />
          <button className="btn btn-send" onClick={() => sendMessage()}>
            Send
          </button>
        </div>

        <div
          style={{
            padding: 12,
            display: "flex",
            justifyContent: "center",
            background: "#343541",
            borderTop: "1px solid rgba(255,255,255,0.04)",
          }}
        >
          {!recording ? (
            <button className="btn btn-record" onClick={startRecording}>
              🎤 Start Recording
            </button>
          ) : (
            <button
              className="btn"
              style={{ background: "#6b7280", color: "#fff" }}
              onClick={stopRecording}
            >
              ⏹ Stop Recording
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
