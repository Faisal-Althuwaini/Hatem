import React, { useState } from "react";
import { Send } from "lucide-react";
import Bot from "../assets/hatem1.png";
import User from "../assets/User-avatar.png";

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { text: "مرحبًا! كيف يمكنني مساعدتك اليوم؟", sender: "bot" },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessage = { text: input, sender: "user" };
    setMessages([...messages, newMessage]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      });

      const data = await response.json();
      console.log(data);
      setMessages((prev) => [
        ...prev,
        { text: data.full_response, sender: "bot" },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-white md:mx-42">
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex items-end ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {msg.sender === "bot" && (
              <img
                src={Bot}
                alt="Bot Avatar"
                className="w-10 h-10 rounded-full ml-2"
              />
            )}
            <div
              className={`max-w-2xl p-4 rounded-lg shadow-md ${
                msg.sender === "user"
                  ? "bg-cyan-600 text-white"
                  : "bg-gray-200 text-gray-800"
              }`}
            >
              {msg.text}
            </div>
            {msg.sender === "user" && (
              <img
                src={User}
                alt="User Avatar"
                className="w-10 h-10 rounded-full mr-2"
              />
            )}
          </div>
        ))}
        {isTyping && (
          <div className="flex items-end justify-start">
            <img
              src={Bot}
              alt="Bot Avatar"
              className="w-10 h-10 rounded-full mr-2"
            />
            <div className="max-w-2xl p-4 rounded-lg shadow-md bg-gray-200 text-gray-800 self-end">
              يكتب...
            </div>
          </div>
        )}
      </div>
      <div className="p-4 border-t flex items-center bg-white">
        <input
          type="text"
          className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-600 text-right"
          // placeholder="اكتب رسالة..."
          placeholder="البوت ليس متوفر حالياً..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyUp={(e) => e.key === "Enter" && sendMessage()}
          // disabled
        />
        <button
          onClick={sendMessage}
          className="mr-2 bg-cyan-600 text-white p-3 rounded-lg hover:bg-cyan-700"
        >
          <Send size={18} />
        </button>
      </div>
      <div>
        <p className="text-gray-500 text-sm mb-4 flex items-center justify-center mr-5">
          حاتم قد يرتكب بعض الأخطاء، لذا يُفضل التحقق من صحة المعلومات قبل
          الاعتماد عليها.
        </p>
      </div>
    </div>
  );
}
