import React, { useState } from "react";
import "../App.css";

export default function GPA_Calculator() {
  const [subjects, setSubjects] = useState([{ grade: "", hours: "" }]);
  const [prevGPA, setPrevGPA] = useState("");
  const [prevHours, setPrevHours] = useState("");
  const [semesterGPA, setSemesterGPA] = useState(null);
  const [finalGPA, setFinalGPA] = useState(null);

  const addSubject = () => {
    setSubjects([...subjects, { grade: "", hours: "" }]);
  };

  const handleChange = (index, field, value) => {
    const updatedSubjects = [...subjects];
    updatedSubjects[index][field] = value;
    setSubjects(updatedSubjects);
  };

  const calculateGPA = () => {
    let totalPoints = 0;
    let totalHours = 0;

    subjects.forEach(({ grade, hours }) => {
      if (grade && hours) {
        const gpaValue = convertGradeToGPA(grade);
        totalPoints += gpaValue * parseFloat(hours);
        totalHours += parseFloat(hours);
      }
    });

    if (totalHours === 0) return; // تجنب القسمة على صفر

    const newSemesterGPA = totalPoints / totalHours;
    setSemesterGPA(newSemesterGPA.toFixed(3));

    if (prevGPA && prevHours) {
      totalPoints += parseFloat(prevGPA) * parseFloat(prevHours);
      totalHours += parseFloat(prevHours);
    }

    const newGPA = totalPoints / totalHours;
    setFinalGPA(newGPA.toFixed(3));
  };

  const convertGradeToGPA = (grade) => {
    const grades = {
      "A+": 4.0,
      A: 3.75,
      "B+": 3.5,
      B: 3.0,
      "C+": 2.5,
      C: 2.0,
      "D+": 1.5,
      D: 1.0,
      F: 0.0,
    };
    return grades[grade] || 0;
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 pl-9 home_bg">
      <div className="bg-white/45 text-gray-800 p-8 w-full max-w-2xl">
        <h1 className="text-3xl leading-10 font-bold text-center text-cyan-600 mb-6">
          🎓 حساب المعدل التراكمي والفصلي
        </h1>

        {/* إدخال GPA السابق */}
        <div className="mb-4 flex flex-col sm:flex-row sm:space-x-8 ">
          <div className="w-full mb-6 md:mb-0">
            <label className="block text-lg font-medium">المعدل السابق:</label>
            <input
              type="number"
              step="0.01"
              className="w-full p-3 border rounded-lg mt-1"
              value={prevGPA}
              onChange={(e) => setPrevGPA(e.target.value)}
              placeholder="أدخل المعدل السابق"
            />
          </div>
          <div className="w-full ">
            <label className="block text-lg font-medium">
              عدد الساعات السابقة:
            </label>
            <input
              type="number"
              className="w-full p-3 border rounded-lg mt-1"
              value={prevHours}
              onChange={(e) => setPrevHours(e.target.value)}
              placeholder="أدخل عدد الساعات السابقة"
            />
          </div>
        </div>
        <h1 className=" text-xl mb-2">المواد:</h1>
        <hr />
        {/* إدخال بيانات المواد */}
        {subjects.map((subject, index) => (
          <div
            key={index}
            className="mb-4 flex flex-col sm:flex-row sm:space-x-4 mt-4"
          >
            <div className="w-full  mb-6 md:mb-0">
              <label className="block text-lg font-medium">الدرجة:</label>
              <select
                className="w-full p-3 border border-l-8 border-transparent outline rounded-lg mt-1"
                value={subject.grade}
                onChange={(e) => handleChange(index, "grade", e.target.value)}
              >
                <option value="">اختر التقدير</option>
                <option value="A+">A+</option>
                <option value="A">A</option>
                <option value="B+">B+</option>
                <option value="B">B</option>
                <option value="C+">C+</option>
                <option value="C">C</option>
                <option value="D+">D+</option>
                <option value="D">D</option>
                <option value="F">F</option>
              </select>
            </div>
            <div className="w-full">
              <label className="block text-lg font-medium">عدد الساعات:</label>
              <input
                type="number"
                className="w-full p-3 outline rounded-lg mt-1"
                value={subject.hours}
                onChange={(e) => handleChange(index, "hours", e.target.value)}
                placeholder="أدخل عدد الساعات"
              />
            </div>
          </div>
        ))}

        <button
          className="w-full bg-white-600 border py-3 mt-4 rounded-lg hover:bg-cyan-600 hover:text-white transition-all"
          onClick={addSubject}
        >
          ➕ إضافة مادة
        </button>

        <button
          className="w-full bg-cyan-600 text-white py-3 mt-4 rounded-lg hover:bg-cyan-700 transition-all"
          onClick={calculateGPA}
        >
          🧮 احسب المعدل
        </button>

        {semesterGPA !== null && (
          <div className="mt-6 text-center bg-gray-100 p-4 rounded-lg">
            <p className="text-xl font-semibold text-gray-800">
              المعدل الفصلي:{" "}
              <span className="text-cyan-600">{semesterGPA}</span>
            </p>
          </div>
        )}

        {finalGPA !== null && (
          <div className="mt-4 text-center bg-gray-100 p-4 rounded-lg">
            <p className="text-xl font-semibold text-gray-800">
              المعدل التراكمي الجديد:{" "}
              <span className="text-cyan-600">{finalGPA}</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
