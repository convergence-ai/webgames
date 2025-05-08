import confetti from "canvas-confetti";
import { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

declare global {
  interface Navigator {
    userAgentData?: {
      platform: string;
    };
  }
}

export const PASSWORD_KeyComboHard = "HARD_KEY_MASTER_2024";
export const TASK_ID_KeyComboHard = "key-combo-hard";

const KeyComboHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_KeyComboHard);
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());
  const [isComplete, setIsComplete] = useState(false);

  // Define the target key combination
  const comboDisplay = "Control + Alt + Shift + S";

  useEffect(() => {
    const targetCombo = ["Control", "Alt", "Shift", "s"];

    // Clear all pressed keys
    const clearKeys = () => {
      setPressedKeys(new Set());
    };

    // Normalize key names to handle shift combinations
    const normalizeKey = (e: KeyboardEvent) => e.key.toLowerCase();

    const handleKeyDown = (e: KeyboardEvent) => {
      const normalizedKey = normalizeKey(e);
      // Prevent duplicate key down events
      if (!pressedKeys.has(normalizedKey)) {
        const newPressedKeys = new Set([
          ...Array.from(pressedKeys),
          normalizedKey,
        ]);

        // Only prevent default if it matches our target combo
        const pressedArray = Array.from(newPressedKeys).map((k) =>
          k.toLowerCase()
        );
        const targetArray = targetCombo.map((k) => k.toLowerCase());
        const isTargetCombo = targetArray.every((key) =>
          pressedArray.includes(key.toLowerCase())
        );

        if (isTargetCombo) {
          e.preventDefault();
        }

        setPressedKeys(newPressedKeys);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const normalizedKey = normalizeKey(e);
      setPressedKeys((prev) => {
        const newSet = new Set(prev);
        newSet.delete(normalizedKey);
        return newSet;
      });
    };

    // Handle window blur
    const handleBlur = () => {
      clearKeys();
    };

    // Handle visibility change
    const handleVisibilityChange = () => {
      if (document.hidden) {
        clearKeys();
      }
    };

    // Check if the pressed combination matches the target
    const checkCombo = () => {
      const pressedArray = Array.from(pressedKeys).map((k) => k.toLowerCase());
      const targetArray = targetCombo.map((k) => k.toLowerCase());

      // Check if all required keys are pressed
      const hasAllKeys = targetArray.every((key) =>
        pressedArray.includes(key.toLowerCase())
      );

      if (hasAllKeys) {
        if (!isComplete) {
          recordSuccess();
        }
        setIsComplete(true);
        confetti({
          particleCount: 100,
          spread: 70,
          origin: { y: 0.6 },
        });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    // Check the combo whenever pressed keys change
    checkCombo();

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [pressedKeys, recordSuccess, isComplete]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-purple-100 to-blue-100 p-4">
      <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold text-center mb-6 text-purple-600">
          Key Combo Challenge (Hard)
        </h1>

        <div className="text-center mb-8">
          <p className="text-lg mb-4">Press the following combination:</p>
          <div className="text-2xl font-mono bg-gray-100 p-4 rounded-lg mb-4">
            {comboDisplay}
          </div>

          <div className="text-sm text-gray-600">
            Currently pressed: {Array.from(pressedKeys).join(" + ") || "None"}
          </div>
        </div>

        {isComplete && (
          <div className="text-center p-4 bg-green-100 text-green-700 rounded-lg">
            <p className="font-bold mb-2">Congratulations! 🎉</p>
            <p>The secret password is: {PASSWORD_KeyComboHard}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default KeyComboHard;
