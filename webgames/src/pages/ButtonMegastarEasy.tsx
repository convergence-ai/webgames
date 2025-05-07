import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_ButtonMegastarEasy = "buttons-easy";
export const PASSWORD_ButtonMegastarEasy = "EasyClick2024";

const ButtonMegastarEasy: React.FC = () => {
  const [showSecret, setShowSecret] = useState(false);
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ButtonMegastarEasy);
  return (
    <div className="min-h-screen p-8 flex flex-col items-center">
      <div className="mb-8 max-w-xl text-center">
        <h1 className="text-3xl font-bold mb-4">
          Button Megastar Challenge (Easy)
        </h1>
        <p className="text-gray-700 mb-2">
          Welcome to the easy version of Button Megastar! Your goal is simple:
          find the one special button that reveals the password.
        </p>
        <p className="text-gray-700 mb-4">
          There are only a few buttons on this page. One of them is the key!
        </p>
      </div>
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 md:gap-8 lg:gap-12">
        {/* Fewer buttons for the easy version */}
        <div className="flex justify-center">
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-md hover:shadow-lg transition-shadow">
            Not this one
          </button>
        </div>

        <div className="flex justify-center">
          <button className="bg-green-500 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-md hover:shadow-lg transition-shadow">
            Try another
          </button>
        </div>

        {/* The deceptive element - now more obvious */}
        <div className="md:col-span-2 flex justify-center items-center flex-col">
          <div
            onClick={() => {
              if (!showSecret) {
                recordSuccess();
                setShowSecret(true);
              }
            }}
            className="text-xl text-gray-600 p-3 border-2 border-dashed border-gray-400 rounded-md hover:bg-gray-100 cursor-pointer transition-colors"
            style={{ userSelect: "none" }}
          >
            Maybe this one? Click to find out!
          </div>
          {showSecret && (
            <div className="mt-4 py-2 px-4 bg-black text-green-400 font-mono text-lg rounded shadow-xl">
              Congratulations! The password is:{" "}
              <span className="font-bold">{PASSWORD_ButtonMegastarEasy}</span>
            </div>
          )}
        </div>

        <div className="flex justify-center">
          <button className="bg-yellow-500 hover:bg-yellow-700 text-black font-bold py-3 px-6 rounded-lg text-lg shadow-md hover:shadow-lg transition-shadow">
            Nope!
          </button>
        </div>
      </div>
    </div>
  );
};

export default ButtonMegastarEasy;
