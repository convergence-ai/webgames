import React, { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_PixelCopyEasy = "pixel-copy-easy";
export const PASSWORD_PixelCopyEasy = "EasyPixels2024"; // Changed password

interface GridProps {
  grid: boolean[][];
  onToggle?: (row: number, col: number) => void;
  gridSize: number; // Added gridSize prop
}

const Grid: React.FC<GridProps> = ({ grid, onToggle, gridSize }) => {
  return (
    // Use gridSize to make grid responsive
    <div
      className={`grid gap-1 w-[300px] h-[300px]`}
      style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}
    >
      {grid.map((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <div
            key={`${rowIndex}-${colIndex}`}
            className={`w-full h-full border border-gray-300 cursor-pointer ${
              cell ? "bg-black" : "bg-white"
            }`}
            onClick={() => onToggle?.(rowIndex, colIndex)}
          />
        ))
      )}
    </div>
  );
};

const PixelCopyEasy: React.FC = () => {
  // Changed component name
  const { recordSuccess } = useTaskAnalytics(TASK_ID_PixelCopyEasy); // Changed TASK_ID
  const [targetGrid, setTargetGrid] = useState<boolean[][]>([]);
  const [userGrid, setUserGrid] = useState<boolean[][]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const gridSize = 5; // Easy grid size

  // Initialize grids
  useEffect(() => {
    const createEmptyGrid = () =>
      Array(gridSize) // Use gridSize
        .fill(false)
        .map(() => Array(gridSize).fill(false)); // Use gridSize

    // Create a simple dot pattern for easy version
    const createSimplePattern = () => {
      const grid = createEmptyGrid();
      grid[Math.floor(gridSize / 2)][Math.floor(gridSize / 2)] = true; // Center dot
      return grid;
    };

    setTargetGrid(createSimplePattern());
    setUserGrid(createEmptyGrid());
  }, []); // Removed gridSize from dependency array as it's constant within this component

  // Toggle user grid cell
  const handleUserGridToggle = (row: number, col: number) => {
    const newGrid = userGrid.map((r, rowIndex) =>
      rowIndex === row
        ? r.map((cell, colIndex) => (colIndex === col ? !cell : cell))
        : r
    );
    setUserGrid(newGrid);

    // Check if grids match
    const matches = newGrid.every((row, rowIndex) =>
      row.every((cell, colIndex) => cell === targetGrid[rowIndex][colIndex])
    );
    setIsComplete(matches);
    if (matches) {
      recordSuccess();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-400 via-cyan-400 to-blue-400 p-8">
      {" "}
      {/* Changed background color */}
      {/* Instructions Card */}
      <div className="flex justify-center mb-8">
        <div className="bg-white rounded-xl p-6 max-w-2xl shadow-sm">
          <h1 className="text-3xl font-bold text-center mb-4 text-gray-800">
            Pixel Copy Challenge (Easy) {/* Changed title */}
          </h1>
          <p className="text-center text-gray-600">
            Recreate the simple pattern. Click squares in the right grid to
            toggle them. Match the pattern to win!{" "}
            {/* Simplified instructions */}
          </p>
        </div>
      </div>
      {/* Success Message */}
      {isComplete && (
        <div className="flex justify-center mb-8">
          <div className="bg-green-100 border border-green-200 rounded-lg p-6 shadow-lg w-full max-w-2xl">
            <h2 className="text-2xl font-bold text-green-800 mb-2">
              Well Done!
            </h2>
            <p className="text-green-700">
              The password is: {PASSWORD_PixelCopyEasy}{" "}
              {/* Changed password variable */}
            </p>
          </div>
        </div>
      )}
      {/* Grids Container */}
      <div className="flex justify-center gap-8 flex-wrap">
        <div className="bg-white rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-center text-gray-700">
            Target Pattern
          </h2>
          <Grid grid={targetGrid} gridSize={gridSize} /> {/* Pass gridSize */}
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-center text-gray-700">
            Your Pattern
          </h2>
          <Grid
            grid={userGrid}
            onToggle={handleUserGridToggle}
            gridSize={gridSize}
          />{" "}
          {/* Pass gridSize */}
        </div>
      </div>
    </div>
  );
};

export default PixelCopyEasy; // Changed export name
