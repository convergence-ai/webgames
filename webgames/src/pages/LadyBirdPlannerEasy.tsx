import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_LadyBirdPlannerEasy = "LADYBIRD_NAVIGATOR_EASY";
export const TASK_ID_LadyBirdPlannerEasy = "ladybird-easy";

interface Position {
  x: number;
  y: number;
}

interface GridCell {
  isWall: boolean;
}

const MOVE_UP = "⬆️";
const MOVE_DOWN = "⬇️";
const MOVE_LEFT = "⬅️";
const MOVE_RIGHT = "➡️";

const LadyBirdPlannerEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_LadyBirdPlannerEasy);
  const gridSize = 8; // Smaller grid
  const [inputSequence, setInputSequence] = useState<string>("");
  const [isComplete, setIsComplete] = useState(false);
  const [moveStatus, setMoveStatus] = useState<"none" | "valid" | "invalid">(
    "none"
  );
  // Reduced and simplified walls for an easier path
  const walls: [number, number][] = [
    [1, 0],
    [1, 1],
    [1, 2],
    [3, 2],
    [3, 3],
    [3, 4],
    [5, 4],
    [5, 5],
    [5, 6],
  ];

  // Initial positions - adjusted for smaller grid
  const ladybirdStart: Position = { x: 0, y: 0 };
  const flowerPosition: Position = { x: 7, y: 7 }; // Target at the opposite corner

  // Define the maze layout
  const initialGrid: GridCell[][] = Array(gridSize)
    .fill(null)
    .map(() => Array(gridSize).fill({ isWall: false }));

  walls.forEach(([x, y]) => {
    if (x < gridSize && y < gridSize) {
      // Ensure walls are within the new grid size
      initialGrid[y][x] = { isWall: true };
    }
  });

  const calculatePath = (sequence: string): Position[] => {
    const path: Position[] = [{ ...ladybirdStart }];
    let currentPos = { ...ladybirdStart };

    const chunkSize = MOVE_DOWN.length;
    const moves: string[] = [];
    for (let i = 0; i < sequence.length; i += chunkSize) {
      moves.push(sequence.slice(i, i + chunkSize));
    }

    for (const move of moves) {
      const newPos = { ...currentPos };

      switch (move) {
        case MOVE_DOWN:
          newPos.y++;
          break;
        case MOVE_UP:
          newPos.y--;
          break;
        case MOVE_LEFT:
          newPos.x--;
          break;
        case MOVE_RIGHT:
          newPos.x++;
          break;
        default:
          console.log("Unknown move:", move);
      }

      if (
        newPos.x < 0 ||
        newPos.x >= gridSize ||
        newPos.y < 0 ||
        newPos.y >= gridSize ||
        walls.some(([wx, wy]) => wx === newPos.x && wy === newPos.y)
      ) {
        return path; // Stop if move is invalid
      }

      path.push({ ...newPos });
      currentPos = { ...newPos };
    }

    return path;
  };

  const handleEmojiClick = (emoji: string) => {
    const newSequence = inputSequence + emoji;
    setInputSequence(newSequence);
  };

  const handleSubmit = () => {
    const path = calculatePath(inputSequence);
    const lastPos = path[path.length - 1];
    if (lastPos.x === flowerPosition.x && lastPos.y === flowerPosition.y) {
      setMoveStatus("valid");
      setIsComplete(true);
      recordSuccess();
    } else {
      setMoveStatus("invalid");
    }
  };

  const handleClear = () => {
    setInputSequence("");
    setIsComplete(false);
    setMoveStatus("none");
  };

  return (
    <div className="w-full min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-2">LadyBird Planner (Easy)</h1>
          <p className="text-gray-300 mb-4">
            Plan the ladybird's path to the flower using directional emojis. The
            ladybird won't move until you submit your solution! This is an
            easier version.
          </p>
        </div>

        {/* Grid */}
        <div className="mb-8 inline-block bg-gray-800 p-2 rounded-lg">
          {initialGrid.map((row, y) => (
            <div key={y} className="flex">
              {row.map((_, x) => (
                <div
                  key={`${x}-${y}`}
                  data-testid="grid-cell"
                  className={`w-10 h-10 border border-gray-700 flex items-center justify-center relative ${
                    // smaller cells
                    walls.some(([wx, wy]) => wx === x && wy === y)
                      ? "bg-gray-700"
                      : ""
                  }`}
                >
                  <span className="text-lg">
                    {" "}
                    {/* smaller emoji */}
                    {x === ladybirdStart.x && y === ladybirdStart.y && "🐞"}
                    {x === flowerPosition.x && y === flowerPosition.y && "🌸"}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Controls */}
        <div className="space-y-4">
          <div className="flex gap-2">
            <button
              onClick={() => handleEmojiClick(MOVE_UP)}
              className="p-2 bg-gray-800 rounded hover:bg-gray-700"
              disabled={isComplete}
            >
              {MOVE_UP}
            </button>
            <button
              onClick={() => handleEmojiClick(MOVE_DOWN)}
              className="p-2 bg-gray-800 rounded hover:bg-gray-700"
              disabled={isComplete}
            >
              {MOVE_DOWN}
            </button>
            <button
              onClick={() => handleEmojiClick(MOVE_LEFT)}
              className="p-2 bg-gray-800 rounded hover:bg-gray-700"
              disabled={isComplete}
            >
              {MOVE_LEFT}
            </button>
            <button
              onClick={() => handleEmojiClick(MOVE_RIGHT)}
              className="p-2 bg-gray-800 rounded hover:bg-gray-700"
              disabled={isComplete}
            >
              {MOVE_RIGHT}
            </button>
            <button
              onClick={handleClear}
              className="px-4 py-2 bg-red-800 rounded hover:bg-red-700"
            >
              Clear
            </button>
            <button
              onClick={handleSubmit}
              className="px-4 py-2 bg-green-800 rounded hover:bg-green-700"
              disabled={isComplete || !inputSequence}
            >
              Submit
            </button>
          </div>

          <div
            className={`text-xl min-h-[2.5rem] p-3 rounded-lg ${
              // smaller text area
              moveStatus === "valid" ? "bg-green-800" : "bg-gray-800"
            }`}
          >
            {inputSequence || "Enter your moves..."}
            {moveStatus === "invalid" && (
              <div className="text-sm mt-2">
                Path does not reach the flower. Try again!
              </div>
            )}
          </div>

          {isComplete && (
            <div className="p-6 bg-green-800 rounded-lg">
              <p className="font-bold text-xl mb-2">Perfect Path Found!</p>
              <p>The password is: {PASSWORD_LadyBirdPlannerEasy}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LadyBirdPlannerEasy;
