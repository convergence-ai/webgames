import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface Disk {
  size: number;
  color: string;
}

type Peg = Disk[];

export const TASK_ID_TowersOfHanoiEasy = "towers-of-hanoi-easy";
export const PASSWORD_TowersOfHanoiEasy = "RecursionMasterTowerEasy"; // Changed password

const TowersOfHanoiEasy = () => {
  // Renamed component
  const colors = [
    "#4A90E2", // Steel Blue - Kept 5 colors, will only use 3
    "#50C878", // Emerald
    "#9B59B6", // Amethyst
    "#F39C12", // Sunflower
    "#E74C3C", // Coral
  ];
  const numDisks = 3; // Easy version: 3 disks

  // Initialize pegs with a proper initial state
  const [pegs, setPegs] = useState<Peg[]>([
    // First peg with all disks
    Array.from({ length: numDisks }, (_, i) => ({
      size: numDisks - i,
      color: colors[numDisks - 1 - i], // Adjusted color indexing if needed
    })),
    [], // Second peg (empty)
    [], // Third peg (empty)
  ]);

  const { recordSuccess } = useTaskAnalytics(TASK_ID_TowersOfHanoiEasy); // Use new TASK_ID
  const [showPassword, setShowPassword] = useState(false);
  const [selectedPeg, setSelectedPeg] = useState<number | null>(null);
  const [moves, setMoves] = useState(0);

  const isValidMove = (fromPeg: number, toPeg: number): boolean => {
    if (pegs[fromPeg].length === 0) return false;
    if (pegs[toPeg].length === 0) return true;
    const fromDisk = pegs[fromPeg][pegs[fromPeg].length - 1];
    const toDisk = pegs[toPeg][pegs[toPeg].length - 1];
    return fromDisk.size < toDisk.size;
  };

  const handlePegClick = (pegIndex: number) => {
    if (selectedPeg === null) {
      if (pegs[pegIndex].length > 0) {
        setSelectedPeg(pegIndex);
      }
    } else {
      if (pegIndex !== selectedPeg && isValidMove(selectedPeg, pegIndex)) {
        setPegs((currentPegs) => {
          const newPegs = currentPegs.map((peg) => [...peg]);
          const disk = newPegs[selectedPeg].pop();
          if (disk) {
            newPegs[pegIndex].push(disk);
          }
          return newPegs;
        });
        setMoves((prev) => prev + 1);

        // Check win condition
        // Win if all disks are on the 3rd peg (index 2)
        // And the number of disks on that peg equals numDisks
        if (pegIndex === 2 && pegs[2].length + 1 === numDisks) {
          // Check after move
          setShowPassword(true);
          recordSuccess();
        }
      }
      setSelectedPeg(null);
    }
  };

  const renderPeg = (pegIndex: number) => {
    return (
      <div className="flex flex-col items-center">
        <div
          className={`h-48 flex flex-col-reverse items-center mb-2 relative ${
            selectedPeg === pegIndex ? "bg-gray-100" : ""
          }`}
          style={{ width: "200px" }} // Peg width can remain same
        >
          {pegs[pegIndex].map((disk, index) => (
            <div
              key={disk.size}
              className={`absolute transition-all duration-300`}
              style={{
                width: `${disk.size * 30}px`, // Disk width scaling
                height: "20px",
                backgroundColor: disk.color,
                bottom: `${index * 24}px`, // Disk stacking
                borderRadius: "8px",
                zIndex: "10",
                boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
              }}
            />
          ))}
          <div
            className="absolute h-full w-2 bg-gray-700 rounded-full"
            style={{ bottom: "0", zIndex: "1" }}
          />
        </div>
        <button
          onClick={() => handlePegClick(pegIndex)}
          className={`px-4 py-2 rounded ${
            selectedPeg === pegIndex
              ? "bg-blue-500 text-white"
              : "bg-gray-200 hover:bg-gray-300"
          }`}
        >
          {selectedPeg === pegIndex ? "Selected" : "Select"}
        </button>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">
          Towers of Hanoi (Easy - 3 Disks)
        </h2>{" "}
        {/* Updated title */}
        <div className="mb-4">
          <p className="text-gray-700">
            Move all the disks from the leftmost peg to the rightmost peg.
            Rules:
            <ul className="list-disc list-inside">
              <li>Only one disk can be moved at a time</li>
              <li>A larger disk cannot be placed on top of a smaller disk</li>
              <li>
                Click a peg to select it, then click another peg to move the top
                disk
              </li>
            </ul>
          </p>
        </div>
        <div className="flex justify-between items-end mb-8 mt-12">
          {renderPeg(0)}
          {renderPeg(1)}
          {renderPeg(2)}
        </div>
        <div className="text-center text-lg font-semibold">Moves: {moves}</div>
        {showPassword && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded">
            <p className="text-green-800 font-medium">
              Congratulations! Secret Password: {PASSWORD_TowersOfHanoiEasy}{" "}
              {/* Use new PASSWORD */}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TowersOfHanoiEasy; // Export renamed component
