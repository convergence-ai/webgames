import { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_ClickCubedHard = "click-cubed-hard";
export const PASSWORD_ClickCubedHard = "TripleClickChampionHard";

const ClickCubedHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ClickCubedHard);
  const [gameStarted, setGameStarted] = useState(false);
  const [button1Clicked, setButton1Clicked] = useState(false);
  const [button2Clicked, setButton2Clicked] = useState(false);
  const [timeLeft, setTimeLeft] = useState(3); // Decreased time limit
  const [gameWon, setGameWon] = useState(false);
  const [gameLost, setGameLost] = useState(false);

  useEffect(() => {
    let timer: number | undefined;
    if (gameStarted && timeLeft > 0) {
      timer = window.setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            clearInterval(timer);
            if (!gameWon) {
              setGameLost(true);
            }
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [gameStarted, timeLeft, gameWon]);

  const handleFirstClick = () => {
    setGameStarted(true);
    setButton1Clicked(true);
  };

  const handleSecondClick = () => {
    setButton2Clicked(true);
  };

  const handleThirdClick = () => {
    setGameWon(true);
  };

  const resetGame = () => {
    setGameStarted(false);
    setButton1Clicked(false);
    setButton2Clicked(false);
    setTimeLeft(3); // Reset to decreased time limit
    setGameWon(false);
    setGameLost(false);
  };

  useEffect(() => {
    if (gameWon) {
      recordSuccess();
    }
  }, [gameWon, recordSuccess]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-4xl font-bold mb-8">Click³ Challenge (Hard)</h1>

      <div className="text-xl mb-8">
        {!gameStarted
          ? "Click the first button to start! You'll have 3 seconds to click all three buttons."
          : `Time left: ${timeLeft}s`}
      </div>

      {gameWon ? (
        <div className="text-center">
          <p className="text-2xl text-green-600 mb-4">🎉 You won! 🎉</p>
          <p className="text-xl mb-4">
            Secret password:{" "}
            <span className="font-mono bg-gray-100 p-2 rounded">
              {PASSWORD_ClickCubedHard}
            </span>
          </p>
          <button
            onClick={resetGame}
            className="bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 transition-colors"
          >
            Play Again
          </button>
        </div>
      ) : gameLost ? (
        <div className="text-center">
          <p className="text-2xl text-red-600 mb-4">Time\'s up! Try again.</p>
          <button
            onClick={resetGame}
            className="bg-red-500 text-white px-6 py-3 rounded-lg hover:bg-red-600 transition-colors"
          >
            Try Again
          </button>
        </div>
      ) : (
        <div className="relative w-80 h-80">
          {/* First button at the top */}
          {!button1Clicked && (
            <button
              onClick={handleFirstClick}
              className="absolute top-0 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 transition-colors"
            >
              Click First!
            </button>
          )}

          {/* Second button bottom left */}
          {button1Clicked && !button2Clicked && (
            <button
              onClick={handleSecondClick}
              className="absolute bottom-0 left-0 bg-purple-500 text-white px-6 py-3 rounded-lg hover:bg-purple-600 transition-colors"
            >
              Click Second!
            </button>
          )}

          {/* Third button bottom right */}
          {button2Clicked && !gameWon && (
            <button
              onClick={handleThirdClick}
              className="absolute bottom-0 right-0 bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 transition-colors"
            >
              Click Third!
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default ClickCubedHard;
