import React from "react";
import CardItem from "./CardItem";
import IdeaImage from "../assets/idea-hatem.webp";
// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";

export default function Resources() {
  return (
    <div className="flex justify-center items-center min-h-screen flex-col pl-4 md:pl-32 md:px-32 home_bg">
      <div className="flex flex-col md:flex-row justify-center items-center space-y-6 md:space-y-0 md:space-x-12 w-xs md:w-full ">
        <motion.img
          src={IdeaImage}
          alt="Hello"
          className="w-48 md:w-64"
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />

        <div className="text-center md:text-right">
          <h1 className=" text-3xl md:text-4xl leading-snug md:mt-8 text-cyan-600 font-bold">
            مصادر مفيدة
          </h1>
          <p className="text-gray-700 mt-4 text-sm md:text-base leading-relaxed">
            موارد تدعم الطالب الأكاديمي في مسيرته التعليمية والمهنية، وتساعده
            على تحقيق أهدافه بفعالية! 📚✨
          </p>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-8 mt-14 grid-cols-1  place-items-center ">
        <CardItem
          title="قوالب سي في جاهزة"
          content="نماذج احترافية تسهّل عليك إنشاء سيرتك الذاتية بسرعة وسهولة، مما يساعدك في التقديم للوظائف بثقة واحترافية! ✨📄"
          url="https://drive.google.com/drive/folders/1Kwq0E241RvmO_bIpvDkhHB7h3Tf4GvFJ?usp=sharing"
          target="_blank"
        />
        <CardItem
          title="كورس لنكدان"
          content="يساعدك على بناء ملف شخصي احترافي، إبراز مهاراتك، والتواصل مع الخبراء بذكاء لزيادة فرصك الوظيفية! 🚀💼"
          url="https://www.youtube.com/watch?v=7JysIkTyccs"
          target="_blank"
        />
        <CardItem
          title="عَـتـبَـة"
          content="منصة عتبة تجمع لك برامج تطوير الخريجين والتدريب التعاوني في مكان واحد، لتسهيل وصولك للفرص المهنية المناسبة! 🚀"
          url="https://go.3atabah.com/dl/d0a5f4"
          target="_blank"
        />
      </div>
    </div>
  );
}
