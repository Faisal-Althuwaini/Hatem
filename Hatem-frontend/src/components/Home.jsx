import HelloImage from "../assets/hello-hatem.png";
import CardItem from "../components/CardItem";
import "../App.css";
// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
export default function Home() {
  const textWithoutEmoji = "اهلاً, انا حاتم مساعدك الأكاديمي"; // Exclude emoji from typing
  const emoji = " 🎓📚"; // Separate emojis
  const [displayText, setDisplayText] = useState("");
  const [typingComplete, setTypingComplete] = useState(false);
  const [cursorVisible, setCursorVisible] = useState(true);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < textWithoutEmoji.length) {
        setDisplayText(() => textWithoutEmoji.slice(0, index + 1));
        index++;
      } else {
        setTypingComplete(true); // Typing is complete, now add emoji
        clearInterval(interval);
      }
    }, 100);

    return () => clearInterval(interval);
  }, []);

  // تأثير وميض المؤشر
  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setCursorVisible((prev) => !prev);
    }, 500);
    return () => clearInterval(cursorInterval);
  }, []);

  return (
    <div
      className={
        "flex justify-center items-center min-h-screen flex-col px-32 home_bg"
      }
    >
      <div className="flex flex-col md:flex-row justify-center items-center space-y-6 md:space-y-0 md:space-x-12 ">
        <motion.img
          src={HelloImage}
          alt="Hello"
          className="w-48 md:w-64"
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />

        <div className="text-center md:text-right">
          <h1 className=" text-3xl md:text-4xl leading-snug md:mt-8 text-cyan-600 font-bold">
            {displayText}
            {typingComplete && emoji}{" "}
            {/* Only show emoji after typing is done */}
            <span className="text-black">{cursorVisible ? "|" : ""}</span>
          </h1>
          <p className="text-gray-700 mt-4 text-sm md:text-base leading-relaxed">
            أجاوب على أسئلتك الجامعية، وأساعدك في حساب معدلك ومعرفة عدد الأيام
            المسموح بها للغياب لتجنب الحرمان.
          </p>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-8 mt-14 grid-cols-1 ">
        <CardItem
          title="تشات بوت - حاتم"
          content="متدرب على لوائح الجامعة، يجاوب على أسئلتك ويوضح لك القوانين ليسهّل رحلتك الأكاديمية! 🎓"
          url="/chatbot"
        />
        <CardItem
          title="حساب الغياب"
          content="يساعدك على معرفة عدد الأيام المسموح بها للغياب لكل محاضرة، لتجنب الحرمان والتخطيط لحضورك الجامعي بذكاء! ✅"
          url="/calculator"
        />
        <CardItem
          title="حساب المعدل"
          content="يحسب معدلك الفصلي والتراكمي بدقة، ويساعدك في معرفة تأثير درجاتك على مستواك الأكاديمي! 🔢📊"
          url="/gpacalc"
        />
      </div>
    </div>
  );
}
