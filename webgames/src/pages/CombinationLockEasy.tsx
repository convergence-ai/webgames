import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_CombinationLockEasy = "combination-lock-easy";
export const PASSWORD_CombinationLockEasy = "COMBO_EASY_2024";

const NOTE_TEXT = `Dear aspiring treasure hunter,

To unlock this simpler treasure, solve these two clues:

1. The first number is how many wheels are on a standard bicycle.

2. The second number is how many days are in a week.

Good luck!
Grampa's Friend`;

const CombinationLockEasy: React.FC = () => {
  const [combination, setCombination] = useState(["00", "00"]);
  const [isUnlocked, setIsUnlocked] = useState(false);
  const { recordSuccess } = useTaskAnalytics(TASK_ID_CombinationLockEasy);

  // The correct combination based on the riddles:
  // 1. Bicycle wheels = 02
  // 2. Days in a week = 07
  const correctCombination = ["02", "07"];

  const handleDownload = () => {
    const blob = new Blob([NOTE_TEXT], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "easy_grampa_note.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCombinationChange = (index: number, value: string) => {
    const newCombination = [...combination];
    newCombination[index] = value;
    setCombination(newCombination);

    const isCorrect = newCombination.every(
      (num, i) => num === correctCombination[i]
    );
    if (isCorrect) {
      recordSuccess();
    }
    setIsUnlocked(isCorrect);
  };

  const generateOptions = () => {
    const options = [];
    for (let i = 0; i <= 99; i++) {
      const value = i.toString().padStart(2, "0");
      options.push(
        <option key={value} value={value}>
          {value}
        </option>
      );
    }
    return options;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-100 to-sky-200 p-8">
      <div className="max-w-xl mx-auto">
        <div className="bg-white rounded-xl p-6 shadow-lg mb-8">
          <h1 className="text-3xl font-bold text-center mb-4 text-gray-800">
            Grampa's Easy Lock
          </h1>
          <p className="text-center text-gray-600 mb-4">
            Download and read the note to solve the easy riddles!
          </p>
          <div className="flex justify-center">
            <button
              onClick={handleDownload}
              className="bg-sky-100 text-sky-800 hover:bg-sky-200 px-6 py-3 rounded-lg flex items-center gap-2 font-medium transition-colors"
            >
              <span>📝</span>
              Download Easy Note
            </button>
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg">
          <div className="flex justify-center gap-4 mb-6">
            {combination.map((num, index) => (
              <div key={index} className="text-center">
                <label className="block text-sm font-medium text-gray-600 mb-2">
                  Number {index + 1}
                </label>
                <select
                  value={num}
                  onChange={(e) =>
                    handleCombinationChange(index, e.target.value)
                  }
                  className="w-20 h-20 text-3xl text-center bg-gray-100 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-sky-500 appearance-none cursor-pointer"
                  style={{
                    WebkitAppearance: "none",
                    MozAppearance: "none",
                  }}
                >
                  {generateOptions()}
                </select>
              </div>
            ))}
          </div>

          <div className="text-center">
            {isUnlocked ? (
              <div className="bg-green-100 text-green-800 p-4 rounded-lg">
                <p className="text-xl font-bold mb-2">🎉 Unlocked! 🎉</p>
                <p>You've cracked the easy combination!</p>
                <p>
                  The secret password is:{" "}
                  <span className="font-bold">
                    {PASSWORD_CombinationLockEasy}
                  </span>
                </p>
              </div>
            ) : (
              <div className="bg-red-100 text-red-800 p-4 rounded-lg">
                <p>The lock is still sealed...</p>
                <p className="text-sm">Try again!</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CombinationLockEasy;
