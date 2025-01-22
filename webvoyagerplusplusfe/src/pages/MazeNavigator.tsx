import React, { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

export const PASSWORD_MazeNavigator = "MAZE_MASTER_2024";

// Define the winning path and dead ends
const WINNING_PATH = "1/1/2/1";
const DEAD_ENDS = new Set([
  "2",
  "3",
  "1/2",
  "1/3",
  "1/1/1",
  "1/1/3",
  "1/1/2/2",
  "1/1/2/3",
]);

// Messages for different states
const MESSAGES = {
  start: "Choose your first door wisely...",
  progress: "The journey continues...",
  dead_end: "A dead end... Try going back!",
  victory: "Congratulations! You've found the exit!",
};

const MazeNavigator: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Get the current path (excluding /maze/)
  const currentPath = location.pathname
    .replace("/maze/", "")
    .replace(/\/$/, "");

  // Determine if we're at the start, a dead end, or victory
  const isStart = !currentPath;
  const isDeadEnd = DEAD_ENDS.has(currentPath);
  const isVictory = currentPath === WINNING_PATH;

  useEffect(() => {
    // Redirect to start if path is invalid
    if (
      !isStart &&
      !isDeadEnd &&
      !isVictory &&
      !currentPath.match(/^[1-3](\/[1-3])*$/)
    ) {
      navigate("/maze/");
    }
  }, [currentPath, navigate, isStart, isDeadEnd, isVictory]);

  const handleDoorClick = (doorNumber: number) => {
    const newPath = currentPath
      ? `${currentPath}/${doorNumber}`
      : `${doorNumber}`;
    navigate(`/maze/${newPath}`);
  };

  // Get appropriate message
  const getMessage = () => {
    if (isStart) return MESSAGES.start;
    if (isDeadEnd) return MESSAGES.dead_end;
    if (isVictory) return MESSAGES.victory;
    return MESSAGES.progress;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-2xl mx-auto">
        {/* Instructions Card */}
        <div className="bg-gray-800 rounded-xl p-6 mb-8">
          <h1 className="text-3xl font-bold text-center mb-4">The Maze</h1>
          <p className="text-center text-gray-300">{getMessage()}</p>
          {!isStart && (
            <p className="text-center text-gray-500 mt-2 text-sm">
              Use your browser's back button to retrace your steps
            </p>
          )}
        </div>

        {/* Doors or Victory Screen */}
        {!isVictory ? (
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map((doorNumber) => (
              <button
                key={doorNumber}
                onClick={() => handleDoorClick(doorNumber)}
                disabled={isDeadEnd}
                className={`
                  aspect-[9/16] rounded-t-xl border-4 border-gray-600
                  ${
                    isDeadEnd
                      ? "bg-gray-800 cursor-not-allowed opacity-50"
                      : "bg-gray-700 hover:bg-gray-600 hover:border-gray-500 transition-colors cursor-pointer"
                  }
                `}
              >
                <div className="text-4xl mt-4">🚪</div>
                <div className="mt-2">Door {doorNumber}</div>
              </button>
            ))}
          </div>
        ) : (
          <div className="text-center bg-green-900 p-8 rounded-xl">
            <h2 className="text-2xl font-bold mb-4">🎉 You made it! 🎉</h2>
            <p className="text-xl mb-4">The secret password is:</p>
            <p className="font-mono text-2xl bg-green-800 p-4 rounded">
              {PASSWORD_MazeNavigator}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MazeNavigator;
