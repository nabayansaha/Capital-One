import { useState } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Show user message immediately
    const newMessages = [...messages, { type: "human", content: input }];
    setMessages(newMessages);

    try {
      // Call FastAPI backend
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "user1", message: input }),
      });

      const data = await res.json();

      setMessages([
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

      <div className="mt-4 flex w-full max-w-lg">
        <input
          className="flex-1 border rounded-lg p-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          className="ml-2 px-4 py-2 bg-green-500 text-white rounded-lg"
          onClick={sendMessage}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
