import React, { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

export const PASSWORD_MazeNavigatorHard = "MAZE_HARD_MASTER_2024";
export const TASK_ID_MazeNavigatorHard = "maze-hard";

// Define the winning path and dead ends for the hard version
const WINNING_PATH_HARD = "3/1/2/1/3"; // Longer path
const DEAD_ENDS_HARD = new Set([
  "1",
  "2",
  "3/2",
  "3/3",
  "3/1/1",
  "3/1/3",
  "3/1/2/2",
  "3/1/2/3",
  "3/1/2/1/1",
  "3/1/2/1/2",
]);

// Messages for different states
const MESSAGES = {
  start: "Choose your first door wisely... this won't be easy!",
  progress: "The labyrinth deepens...",
  dead_end: "A dead end! The maze tries to trick you. Go back!",
  victory: "Unbelievable! You've conquered the complex maze!",
};

const MazeNavigatorHard: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Get the current path (excluding /maze-hard/)
  const currentPath = location.pathname
    .replace(`/${TASK_ID_MazeNavigatorHard}/`, "")
    .replace(/\/$/, "");

  // Determine if we're at the start, a dead end, or victory
  const isStart = !currentPath;
  const isDeadEnd = DEAD_ENDS_HARD.has(currentPath);
  const isVictory = currentPath === WINNING_PATH_HARD;

  useEffect(() => {
    // Redirect to start if path is invalid
    if (
      !isStart &&
      !isDeadEnd &&
      !isVictory &&
      !currentPath.match(/^[1-3](\/[1-3])*$/) &&
      currentPath !== ""
    ) {
      navigate(`/${TASK_ID_MazeNavigatorHard}/`);
    }
  }, [currentPath, navigate, isStart, isDeadEnd, isVictory]);

  const handleDoorClick = (doorNumber: number) => {
    const newPath = currentPath
      ? `${currentPath}/${doorNumber}`
      : `${doorNumber}`;
    navigate(`/${TASK_ID_MazeNavigatorHard}/${newPath}`);
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
          <h1 className="text-3xl font-bold text-center mb-4">
            The Maze (Hard)
          </h1>
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
          <div className="text-center bg-red-900 p-8 rounded-xl">
            <h2 className="text-2xl font-bold mb-4">
              🎉 You are a Maze Master! 🎉
            </h2>
            <p className="text-xl mb-4">The ultimate secret password is:</p>
            <p className="font-mono text-2xl bg-red-800 p-4 rounded">
              {PASSWORD_MazeNavigatorHard}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MazeNavigatorHard;
