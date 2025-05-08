import confetti from "canvas-confetti";
import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_ButtonHoldHard = "button-hold-hard";
export const PASSWORD_ButtonHoldHard = "HOLD_LONGER_2024"; // Changed password

const ButtonHoldHard = () => {
  // Renamed component
  const [holdTime, setHoldTime] = useState(0);
  const [isHolding, setIsHolding] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const intervalRef = useRef<number>();
  const startTimeRef = useRef<number>(0);
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ButtonHoldHard); // Changed task ID
  useEffect(() => {
    if (isComplete) {
      confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 },
      });
      recordSuccess();
    }
  }, [isComplete, recordSuccess]);

  const startHolding = () => {
    if (isComplete) return;

    setIsHolding(true);
    startTimeRef.current = Date.now();
    intervalRef.current = window.setInterval(() => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      setHoldTime(elapsed);

      if (elapsed >= 5) {
        // Changed hold time to 5 seconds
        stopHolding();
        setIsComplete(true);
      }
    }, 100); // Update every 100ms for smooth timer
  };

  const stopHolding = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    if (!isComplete) {
      setHoldTime(0);
    }
    setIsHolding(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-red-100 to-pink-100 p-4">
      {" "}
      {/* Changed background color slightly for hard */}
      <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold text-center mb-6 text-red-600">
          {" "}
          {/* Changed text color */}
          Button Hold Challenge (Hard) {/* Changed title */}
        </h1>

        <div className="text-center mb-6">
          <p className="text-lg mb-4">
            Hold the button down for five seconds! {/* Changed instruction */}
          </p>
          <p className="text-2xl font-mono">{holdTime.toFixed(1)}s</p>
        </div>

        <div className="flex justify-center mb-6">
          <button
            onMouseDown={startHolding}
            onMouseUp={stopHolding}
            onMouseLeave={stopHolding}
            onTouchStart={startHolding}
            onTouchEnd={stopHolding}
            className={`
              w-48 h-48 rounded-full shadow-lg transition-all duration-200
              ${isHolding ? "scale-95 shadow-sm" : "scale-100"}
              ${
                isComplete
                  ? "bg-green-500 cursor-default"
                  : "bg-pink-500 hover:bg-pink-600 cursor-pointer" // Changed button color
              }
              ${isHolding ? "bg-rose-500" : ""} {/* Changed holding color */}
            `}
          >
            {isComplete ? "🎉" : ""}
          </button>
        </div>

        {isComplete && (
          <div className="text-center p-4 bg-green-100 text-green-700 rounded-lg">
            <p className="font-bold mb-2">Congratulations! 🎉</p>
            <p>The secret password is: {PASSWORD_ButtonHoldHard}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ButtonHoldHard; // Renamed export
