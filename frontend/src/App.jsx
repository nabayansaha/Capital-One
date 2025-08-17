import { useState } from "react";

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
      setMessages([
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
    <div className="flex flex-col items-center p-6 min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-4">🌱 KrishiMitra Chatbot</h1>

      <div className="w-full max-w-lg p-4 bg-white shadow rounded-lg h-[500px] overflow-y-auto">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`my-2 p-2 rounded-lg ${
              m.type === "human"
                ? "bg-green-100 text-right"
                : "bg-blue-100 text-left"
            }`}
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
