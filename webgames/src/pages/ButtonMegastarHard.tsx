import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_ButtonMegastarHard = "buttons-hard";
export const PASSWORD_ButtonMegastarHard = "SuperClickerPro2024";

const ButtonMegastarHard: React.FC = () => {
  const [checked, setChecked] = useState(false);
  const [showSecret, setShowSecret] = useState(false);
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ButtonMegastarHard);

  // Helper to create many similar elements to increase clutter
  const renderDummyElement = (
    text: string,
    color: string,
    type: "button" | "div" | "link" | "select" | "input" | "checkbox" | "radio",
    i: number
  ) => {
    const baseClasses = "font-bold py-2 px-4 rounded m-1";
    const hoverClasses = `hover:opacity-80`;

    switch (type) {
      case "button":
        return (
          <button
            className={`${baseClasses} bg-${color}-500 text-white ${hoverClasses}`}
          >
            {text} {i}
          </button>
        );
      case "div":
        return (
          <div
            className={`${baseClasses} bg-${color}-300 text-${color}-800 cursor-pointer ${hoverClasses}`}
          >
            {text} {i}
          </div>
        );
      case "link":
        return (
          <a
            href="#"
            className={`${baseClasses} bg-${color}-600 text-white ${hoverClasses}`}
          >
            {text} {i}
          </a>
        );
      case "select":
        return (
          <select
            className={`form-select p-2 border rounded hover:border-${color}-500 cursor-pointer m-1 w-32`}
          >
            <option>
              {text} A {i}
            </option>
            <option>
              {text} B {i}
            </option>
          </select>
        );
      case "input":
        return (
          <input
            type="range"
            className={`w-32 cursor-pointer m-1`}
            min="0"
            max="100"
          />
        );
      case "checkbox":
        return (
          <label className="flex items-center space-x-1 cursor-pointer m-1">
            <input
              type="checkbox"
              className={`form-checkbox h-4 w-4 text-${color}-600 cursor-pointer`}
            />
            <span className={`text-gray-700 text-sm`}>
              {text} {i}
            </span>
          </label>
        );
      case "radio":
        return (
          <label className="flex items-center space-x-1 cursor-pointer m-1">
            <input
              type="radio"
              name={`radio-group-${i % 5}`}
              className={`form-radio h-4 w-4 text-${color}-600 cursor-pointer`}
            />
            <span className={`text-gray-700 text-sm`}>
              {text} {i}
            </span>
          </label>
        );
      default:
        return null;
    }
  };

  return (
    <div className="h-screen p-8 overflow-y-auto">
      {" "}
      {/* Made scrollable */}
      <div className="mb-8 max-w-2xl mx-auto text-center">
        <h1 className="text-2xl font-bold mb-4">
          Button Megastar Challenge (Hard)
        </h1>
        <p className="text-gray-600 mb-2">
          Welcome to the Hardcore Button Megastar challenge! This page is
          FLOODED with interactive elements.
        </p>
        <p className="text-gray-600 mb-4">
          Your mission, should you choose to accept it, is to find the ONE
          element that reveals the password. Good luck, you'll need it!
        </p>
      </div>
      <div className="flex flex-wrap justify-center items-center gap-2">
        {/* Generate a lot of dummy elements */}
        {Array.from({ length: 15 }).map((_, i) =>
          renderDummyElement("Dummy", "blue", "button", i)
        )}
        {Array.from({ length: 15 }).map((_, i) =>
          renderDummyElement("Zone", "yellow", "div", i + 15)
        )}
        {Array.from({ length: 15 }).map((_, i) =>
          renderDummyElement("Link", "purple", "link", i + 30)
        )}
        {Array.from({ length: 10 }).map((_, i) =>
          renderDummyElement("Option", "gray", "select", i + 45)
        )}
        {Array.from({ length: 10 }).map((_, i) =>
          renderDummyElement("Slide", "teal", "input", i + 55)
        )}
        {Array.from({ length: 10 }).map((_, i) =>
          renderDummyElement("Check", "pink", "checkbox", i + 65)
        )}
        {Array.from({ length: 10 }).map((_, i) =>
          renderDummyElement("Select", "red", "radio", i + 75)
        )}

        {/* Original elements scattered */}
        <div className="order-1">
          <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded m-1">
            Click Me!
          </button>
        </div>

        <div className="order-10">
          <div className="text-4xl cursor-pointer hover:opacity-80 m-1">💡</div>
        </div>

        <div className="order-5">
          <button className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-full m-1">
            Press Here
          </button>
        </div>

        <div className="order-12">
          <select className="form-select p-2 border rounded hover:border-blue-500 cursor-pointer m-1">
            <option>Choose me</option>
            <option>Or me</option>
            <option>Maybe me?</option>
          </select>
        </div>

        <div className="order-3">
          <input
            type="range"
            className="w-full cursor-pointer m-1"
            min="0"
            max="100"
          />
        </div>

        {/* The deceptive element - hidden amongst many */}
        <div className="order-20 relative m-2 p-2 border border-dashed border-transparent hover:border-gray-400">
          <span
            onClick={() => {
              if (!showSecret) {
                recordSuccess();
                setShowSecret(true);
              }
            }}
            className="text-gray-400 hover:text-gray-800 transition-colors duration-300 text-xs"
            style={{ userSelect: "none" }}
          >
            this is the one
          </span>
          {showSecret && (
            <div className="absolute mt-2 py-1 px-2 bg-black text-green-400 font-mono text-sm rounded opacity-90 transform transition-all duration-300 z-50 whitespace-nowrap -translate-x-1/2 left-1/2">
              Password:{" "}
              <span className="font-bold">{PASSWORD_ButtonMegastarHard}</span>
            </div>
          )}
        </div>

        {Array.from({ length: 15 }).map((_, i) =>
          renderDummyElement("Action", "indigo", "button", i + 85)
        )}
        {Array.from({ length: 15 }).map((_, i) =>
          renderDummyElement("Area", "lime", "div", i + 100)
        )}

        <div className="order-7">
          <a
            href="#"
            className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg m-1"
          >
            Tap This
          </a>
        </div>

        <div className="order-15">
          <label className="flex items-center space-x-2 cursor-pointer m-1">
            <input
              type="checkbox"
              className="form-checkbox h-5 w-5 text-blue-600 cursor-pointer"
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
            />
            <span className="text-gray-700">Check this!</span>
          </label>
        </div>
        {Array.from({ length: 15 }).map((_, i) =>
          renderDummyElement("More", "orange", "link", i + 115)
        )}
        {Array.from({ length: 15 }).map((_, i) =>
          renderDummyElement("Stuff", "cyan", "div", i + 130)
        )}
      </div>
      <div className="h-48"></div>{" "}
      {/* Extra space at the bottom to ensure scrolling */}
    </div>
  );
};

export default ButtonMegastarHard;
