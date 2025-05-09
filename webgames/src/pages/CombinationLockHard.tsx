import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_CombinationLockHard = "combination-lock-hard";
export const PASSWORD_CombinationLockHard = "COMBO_HARD_2024";

const NOTE_TEXT = `Greetings, advanced cryptographer,

Your challenge, should you choose to accept it, involves deciphering a four-part combination.
Only the most astute will succeed. Consider these enigmas:

1. The first number: Combine the total number of sides on a standard die 
   with the count of states in the United States of America.

2. The second number: From the year humanity first set foot on the moon, 
   subtract the year nineteen hundred.
   
3. The third number: How many keys are typically found on a full-sized piano?

4. The fourth number: Seek the atomic number of the element Aurum (Au).

May your intellect be your guide.

A Mysterious Benefactor`;

const CombinationLockHard: React.FC = () => {
  const [combination, setCombination] = useState(["00", "00", "00", "00"]);
  const [isUnlocked, setIsUnlocked] = useState(false);
  const { recordSuccess } = useTaskAnalytics(TASK_ID_CombinationLockHard);

  // The correct combination based on the riddles:
  // 1. Die (6) + US States (50) = 56
  // 2. Moon landing (1969) - 1900 = 69
  // 3. Piano keys = 88
  // 4. Atomic number of Gold (Au) = 79
  const correctCombination = ["56", "69", "88", "79"];

  const handleDownload = () => {
    const blob = new Blob([NOTE_TEXT], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mysterious_note_hard.txt";
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
    <div className="min-h-screen bg-gradient-to-br from-slate-700 to-slate-800 p-8">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-xl p-6 shadow-lg mb-8">
          <h1 className="text-3xl font-bold text-center mb-4 text-gray-800">
            The Cryptographer's Lock
          </h1>
          <p className="text-center text-gray-600 mb-4">
            Download the benefactor's note. Decipher the complex riddles to
            unlock the ultimate combination!
          </p>
          <div className="flex justify-center">
            <button
              onClick={handleDownload}
              className="bg-slate-600 text-white hover:bg-slate-700 px-6 py-3 rounded-lg flex items-center gap-2 font-medium transition-colors"
            >
              <span>📜</span>
              Download Benefactor's Note
            </button>
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg">
          <div className="flex flex-wrap justify-center gap-4 mb-6">
            {combination.map((num, index) => (
              <div key={index} className="text-center">
                <label className="block text-sm font-medium text-gray-600 mb-2">
                  Cipher {index + 1}
                </label>
                <select
                  value={num}
                  onChange={(e) =>
                    handleCombinationChange(index, e.target.value)
                  }
                  className="w-20 h-20 text-3xl text-center bg-gray-100 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-slate-500 appearance-none cursor-pointer"
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
                <p className="text-xl font-bold mb-2">🎉 UNLOCKED! 🎉</p>
                <p>Your intellect has prevailed! The vault is open!</p>
                <p>
                  The master password is:{" "}
                  <span className="font-bold">
                    {PASSWORD_CombinationLockHard}
                  </span>
                </p>
              </div>
            ) : (
              <div className="bg-orange-100 text-orange-800 p-4 rounded-lg">
                <p>The mechanism remains steadfastly locked...</p>
                <p className="text-sm">Re-evaluate your deductions!</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CombinationLockHard;
