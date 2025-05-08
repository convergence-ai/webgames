import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_ColorHarmonyHard = "color-harmony-hard";
export const PASSWORD_ColorHarmonyHard = "chromatic_conqueror_hard"; // Or a new password

interface RGB {
  r: number;
  g: number;
  b: number;
}

// Calculate color difference (simple Euclidean distance)
function getColorDifference(color1: RGB, color2: RGB): number {
  return Math.sqrt(
    Math.pow(color1.r - color2.r, 2) +
      Math.pow(color1.g - color2.g, 2) +
      Math.pow(color1.b - color2.b, 2)
  );
}

const ColorHarmonyHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ColorHarmonyHard);
  const [userColor, setUserColor] = useState<RGB>({ r: 128, g: 128, b: 128 });
  const [targetColor, setTargetColor] = useState<RGB>({ r: 0, g: 0, b: 0 });
  const [gameActive, setGameActive] = useState(false);
  const [hasWon, setHasWon] = useState(false);

  // Generate a random color
  const generateRandomColor = (): RGB => ({
    r: Math.floor(Math.random() * 256),
    g: Math.floor(Math.random() * 256),
    b: Math.floor(Math.random() * 256),
  });

  // Start new game
  const startGame = () => {
    setGameActive(true);
    setHasWon(false);
    setTargetColor(generateRandomColor());
    setUserColor({ r: 128, g: 128, b: 128 });
  };

  const checkMatch = () => {
    if (gameActive && !hasWon) {
      const difference = getColorDifference(userColor, targetColor);
      if (difference < 15) {
        // Decreased threshold for harder matching
        setHasWon(true);
        setGameActive(false);
        recordSuccess();
      }
    }
  };

  const rgbToHex = (rgb: RGB): string => {
    return `#${rgb.r.toString(16).padStart(2, "0")}${rgb.g
      .toString(16)
      .padStart(2, "0")}${rgb.b.toString(16).padStart(2, "0")}`;
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
        {!gameActive && !hasWon ? (
          <div className="text-center">
            <h1 className="text-2xl font-bold mb-4">Color Harmony (Hard)</h1>
            <p className="mb-4">
              Match the target color using the RGB sliders! This one requires
              precision.
            </p>
            <button
              onClick={startGame}
              className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
            >
              Start Game
            </button>
          </div>
        ) : hasWon ? (
          <div className="text-center">
            <h1 className="text-2xl font-bold mb-4">Impressive!</h1>
            <p className="mb-4">You matched the color perfectly!</p>
            <p className="mb-4">
              The secret password is:{" "}
              <span className="font-bold">{PASSWORD_ColorHarmonyHard}</span>
            </p>
            <button
              onClick={startGame}
              className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
            >
              Play Again
            </button>
          </div>
        ) : (
          <>
            <div className="flex justify-between mb-6">
              <div className="text-center">
                <div
                  className="w-20 h-20 rounded border-2 border-gray-300"
                  style={{ backgroundColor: rgbToHex(userColor) }}
                />
                <p className="mt-1">Your Color</p>
              </div>
              <div className="text-center">
                <div
                  className="w-20 h-20 rounded border-2 border-gray-300"
                  style={{ backgroundColor: rgbToHex(targetColor) }}
                />
                <p className="mt-1">Target Color</p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Red
                </label>
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={userColor.r}
                  onChange={(e) =>
                    setUserColor({ ...userColor, r: parseInt(e.target.value) })
                  }
                  onMouseUp={checkMatch}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Green
                </label>
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={userColor.g}
                  onChange={(e) =>
                    setUserColor({ ...userColor, g: parseInt(e.target.value) })
                  }
                  onMouseUp={checkMatch}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Blue
                </label>
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={userColor.b}
                  onChange={(e) =>
                    setUserColor({ ...userColor, b: parseInt(e.target.value) })
                  }
                  onMouseUp={checkMatch}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ColorHarmonyHard;
