import React, { useState } from "react";
import "../App.css";

export default function AbsenceCalculator() {
  const [hoursPerWeek, setHoursPerWeek] = useState("");
  const [lecturesPerWeek, setLecturesPerWeek] = useState("");
  const [allowedAbsences, setAllowedAbsences] = useState(null);

  const calculateAbsence = () => {
    if (!hoursPerWeek || !lecturesPerWeek) return;

    const weeks = 16; // عدد أسابيع الفصل
    let totalLectures = lecturesPerWeek * weeks; // إجمالي عدد المحاضرات في الفصل

    // // التعامل مع المواد ذات 3 ساعات (تقسيمها 2+1)
    // if (parseInt(hoursPerWeek) === 3 && parseInt(lecturesPerWeek) === 2) {
    //   totalLectures = 32; // فرض أن المادة توزع 2+1
    // }

    const maxAbsences = Math.floor(totalLectures * 0.22); // الغيابات المسموحة

    setAllowedAbsences(maxAbsences);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center home_bg p-6">
      <div className="bg-white/45 text-gray-800 p-6 rounded-lg  w-full max-w-xl">
        <h1 className="text-3xl leading-10 font-bold text-center text-cyan-600 mb-6">
          📝 حساب الغيابات المسموحة
        </h1>

        <div className="mb-4">
          <label className="block text-lg font-medium">
            عدد الساعات الأسبوعية للمادة:
          </label>
          <input
            type="number"
            className="w-full p-3 border rounded-lg mt-1"
            value={hoursPerWeek}
            onChange={(e) => setHoursPerWeek(e.target.value)}
            placeholder="أدخل عدد الساعات"
          />
        </div>

        <div className="mb-4">
          <label className="block text-lg font-medium">
            عدد المحاضرات الأسبوعية:
          </label>
          <input
            type="number"
            className="w-full p-3 border rounded-lg mt-1"
            value={lecturesPerWeek}
            onChange={(e) => setLecturesPerWeek(e.target.value)}
            placeholder="أدخل عدد المحاضرات"
          />
        </div>

        <button
          className="w-full bg-cyan-600 text-white p-3 rounded-lg hover:bg-cyan-700 transition"
          onClick={calculateAbsence}
        >
          احسب الغياب
        </button>

        {allowedAbsences !== null && (
          <div className="mt-4 text-center">
            <p className="text-lg font-semibold">
              الغيابات المسموحة:{" "}
              <span className="text-cyan-600">{allowedAbsences} محاضرات</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
