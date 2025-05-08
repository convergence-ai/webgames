import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface Disk {
  size: number;
  color: string;
}

type Peg = Disk[];

export const TASK_ID_TowersOfHanoiHard = "towers-of-hanoi-hard";
export const PASSWORD_TowersOfHanoiHard = "RecursionMasterTowerHard"; // Changed password

const TowersOfHanoiHard = () => {
  // Renamed component
  const colors = [
    "#4A90E2", // Steel Blue
    "#50C878", // Emerald
    "#9B59B6", // Amethyst
    "#F39C12", // Sunflower
    "#E74C3C", // Coral
    "#34495E", // Wet Asphalt - Added a 6th color
  ];
  const numDisks = 6; // Hard version: 6 disks
  const minMoves = Math.pow(2, numDisks) - 1;

  // Initialize pegs with a proper initial state
  const [pegs, setPegs] = useState<Peg[]>([
    // First peg with all disks
    Array.from({ length: numDisks }, (_, i) => ({
      size: numDisks - i,
      color: colors[numDisks - 1 - i],
    })),
    [], // Second peg (empty)
    [], // Third peg (empty)
  ]);

  const { recordSuccess } = useTaskAnalytics(TASK_ID_TowersOfHanoiHard); // Use new TASK_ID
  const [showPassword, setShowPassword] = useState(false);
  const [selectedPeg, setSelectedPeg] = useState<number | null>(null);
  const [moves, setMoves] = useState(0);
  const [message, setMessage] = useState<string>("");

  const isValidMove = (fromPeg: number, toPeg: number): boolean => {
    if (pegs[fromPeg].length === 0) return false;
    if (pegs[toPeg].length === 0) return true;
    const fromDisk = pegs[fromPeg][pegs[fromPeg].length - 1];
    const toDisk = pegs[toPeg][pegs[toPeg].length - 1];
    return fromDisk.size < toDisk.size;
  };

  const handlePegClick = (pegIndex: number) => {
    if (showPassword) return; // Don't allow moves after solving

    if (selectedPeg === null) {
      if (pegs[pegIndex].length > 0) {
        setSelectedPeg(pegIndex);
      }
    } else {
      if (pegIndex !== selectedPeg && isValidMove(selectedPeg, pegIndex)) {
        const newMoves = moves + 1;
        setMoves(newMoves);

        setPegs((currentPegs) => {
          const newPegs = currentPegs.map((peg) => [...peg]);
          const disk = newPegs[selectedPeg].pop();
          if (disk) {
            newPegs[pegIndex].push(disk);
          }
          return newPegs;
        });

        // Check win condition
        if (pegIndex === 2 && pegs[2].length + 1 === numDisks) {
          // Check after move logic (pegs state is not updated yet here)
          // Need to check the state of the board *after* the current move is applied
          // The `pegs` state used in the condition `pegs[2].length + 1` refers to the state *before* the current move.
          // We can construct the next state of the target peg to check accurately.
          const targetPegAfterMove = [
            ...pegs[pegIndex],
            pegs[selectedPeg][pegs[selectedPeg].length - 1],
          ];

          if (targetPegAfterMove.length === numDisks) {
            // All disks on the target peg
            if (newMoves === minMoves) {
              setShowPassword(true);
              recordSuccess();
              setMessage(`Solved in the minimum ${minMoves} moves!`);
            } else if (newMoves < minMoves) {
              // This case should not happen if solved correctly, but good for robustness
              setMessage(
                `Almost there! Solved, but not in the minimum moves. Moves: ${newMoves}, Minimum: ${minMoves}`
              );
              // Potentially don't show password or record success if not minimum.
            } else {
              setMessage(
                `Solved, but you took ${newMoves} moves. The minimum is ${minMoves}. Try again for the password!`
              );
              // Reset game or indicate failure to meet minimum moves
            }
          }
        }
      }
      setSelectedPeg(null);
    }
  };

  const renderPeg = (pegIndex: number) => {
    return (
      <div className="flex flex-col items-center">
        <div
          className={`h-60 flex flex-col-reverse items-center mb-2 relative ${
            selectedPeg === pegIndex ? "bg-gray-100" : ""
          }`}
          style={{ width: "240px" }} // Slightly wider for more disks
        >
          {pegs[pegIndex].map((disk, index) => (
            <div
              key={disk.size}
              className={`absolute transition-all duration-300`}
              style={{
                width: `${disk.size * 25}px`, // Adjusted disk width scaling
                height: "20px",
                backgroundColor: disk.color,
                bottom: `${index * 22}px`, // Adjusted disk stacking
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
          disabled={showPassword}
        >
          {selectedPeg === pegIndex ? "Selected" : "Select"}
        </button>
      </div>
    );
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      {" "}
      {/* Wider container */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">
          Towers of Hanoi (Hard - {numDisks} Disks)
        </h2>
        <p className="text-gray-600 mb-2">Minimum moves required: {minMoves}</p>
        <div className="mb-4">
          <p className="text-gray-700">
            Move all the disks from the leftmost peg to the rightmost peg in the
            minimum number of moves. Rules:
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
        {message && (
          <div
            className={`mt-4 p-3 rounded text-center ${
              showPassword
                ? "bg-green-100 text-green-700"
                : "bg-yellow-100 text-yellow-700"
            }`}
          >
            {message}
          </div>
        )}
        {showPassword && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded">
            <p className="text-green-800 font-medium">
              Congratulations! Secret Password: {PASSWORD_TowersOfHanoiHard}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TowersOfHanoiHard;
