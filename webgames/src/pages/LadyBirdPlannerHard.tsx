import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_LadyBirdPlannerHard = "LADYBIRD_NAVIGATOR_HARD";
export const TASK_ID_LadyBirdPlannerHard = "ladybird-hard";

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

const LadyBirdPlannerHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_LadyBirdPlannerHard);
  const gridSize = 15; // Larger grid
  const [inputSequence, setInputSequence] = useState<string>("");
  const [isComplete, setIsComplete] = useState(false);
  const [moveStatus, setMoveStatus] = useState<"none" | "valid" | "invalid">(
    "none"
  );
  // More complex and numerous walls for a harder path
  const walls: [number, number][] = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [10, 0],
    [11, 0],
    [12, 0],
    [13, 0],
    [14, 0],
    [0, 1],
    [14, 1],
    [0, 2],
    [2, 2],
    [3, 2],
    [4, 2],
    [6, 2],
    [7, 2],
    [8, 2],
    [10, 2],
    [11, 2],
    [12, 2],
    [14, 2],
    [0, 3],
    [2, 3],
    [6, 3],
    [10, 3],
    [14, 3],
    [0, 4],
    [2, 4],
    [3, 4],
    [4, 4],
    [5, 4],
    [6, 4],
    [8, 4],
    [9, 4],
    [10, 4],
    [12, 4],
    [13, 4],
    [14, 4],
    [0, 5],
    [8, 5],
    [12, 5],
    [14, 5],
    [0, 6],
    [1, 6],
    [2, 6],
    [3, 6],
    [4, 6],
    [6, 6],
    [7, 6],
    [8, 6],
    [10, 6],
    [11, 6],
    [12, 6],
    [14, 6],
    [0, 7],
    [4, 7],
    [6, 7],
    [10, 7],
    [14, 7],
    [0, 8],
    [1, 8],
    [2, 8],
    [4, 8],
    [6, 8],
    [7, 8],
    [8, 8],
    [9, 8],
    [10, 8],
    [12, 8],
    [13, 8],
    [14, 8],
    [0, 9],
    [2, 9],
    [4, 9],
    [12, 9],
    [14, 9],
    [0, 10],
    [2, 10],
    [3, 10],
    [4, 10],
    // [5, 10],
    [6, 10],
    [7, 10],
    [8, 10],
    [10, 10],
    [11, 10],
    [12, 10],
    [14, 10],
    [0, 11],
    [8, 11],
    // [10, 11],
    [14, 11],
    [0, 12],
    [1, 12],
    [2, 12],
    [3, 12],
    [4, 12],
    [5, 12],
    [6, 12],
    // [8, 12],
    [10, 12],
    [12, 12],
    // [13, 12],
    [14, 12],
    [0, 13],
    [6, 13],
    [8, 13],
    [10, 13],
    [12, 13],
    [14, 13],
    [0, 14],
    [1, 14],
    [2, 14],
    [3, 14],
    [4, 14],
    [5, 14],
    [6, 14],
    [7, 14],
    [8, 14],
    [9, 14],
    [10, 14],
    [11, 14],
    [12, 14],
    [13, 14],
    [14, 14],
  ];

  // Initial positions
  const ladybirdStart: Position = { x: 1, y: 1 }; // Start inside the border
  const flowerPosition: Position = { x: 13, y: 13 }; // Target inside the border

  const initialGrid: GridCell[][] = Array(gridSize)
    .fill(null)
    .map(() => Array(gridSize).fill({ isWall: false }));

  walls.forEach(([x, y]) => {
    initialGrid[y][x] = { isWall: true };
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
        return path;
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
      <div className="max-w-5xl mx-auto">
        {" "}
        {/* Increased max-width for larger grid */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-2">LadyBird Planner (Hard)</h1>
          <p className="text-gray-300 mb-4">
            Plan the ladybird's path to the flower using directional emojis. The
            ladybird won't move until you submit your solution! This is a harder
            version with a larger maze.
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
                  className={`w-8 h-8 border border-gray-700 flex items-center justify-center relative ${
                    // smaller cells for larger grid
                    walls.some(([wx, wy]) => wx === x && wy === y)
                      ? "bg-gray-700"
                      : ""
                  }`}
                >
                  <span className="text-sm">
                    {" "}
                    {/* smaller emoji for smaller cells */}
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
            className={`text-2xl min-h-[3rem] p-4 rounded-lg ${
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
              <p>The password is: {PASSWORD_LadyBirdPlannerHard}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LadyBirdPlannerHard;
