import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_RightClickRevealHard = "HARD_CONTEXT_CONNOISSEUR";
export const TASK_ID_RightClickRevealHard = "right-click-hard";

const RightClickRevealHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_RightClickRevealHard);
  const [password, setPassword] = useState("");
  const [revealed, setRevealed] = useState(false);

  // Decoy area
  const handleDecoyContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    alert("Nice try! But this isn't the spot. Keep looking!");
  };

  // Actual target area
  const handleTargetContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    if (!revealed) {
      setPassword(PASSWORD_RightClickRevealHard);
      setRevealed(true);
      recordSuccess();
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Right Click Challenge (Hard)</h1>

      <p className="mb-6">
        This one is trickier! There are multiple areas. Only a specific small,
        hidden spot will reveal the true password when right-clicked. Don't get
        fooled by decoys!
      </p>

      <div className="grid grid-cols-3 gap-4">
        <div
          className="h-32 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center cursor-context-menu hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          onContextMenu={handleDecoyContextMenu}
        >
          Decoy 1
        </div>
        <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center cursor-context-menu hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors relative">
          Decoy 2{/* The actual target, small and somewhat hidden */}
          <div
            title="Right click me!"
            className="absolute bottom-1 right-1 w-8 h-8 bg-blue-500 opacity-50 hover:opacity-100 rounded-full cursor-context-menu"
            onContextMenu={handleTargetContextMenu}
          />
        </div>
        <div
          className="h-32 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center cursor-context-menu hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          onContextMenu={handleDecoyContextMenu}
        >
          Decoy 3
        </div>
      </div>

      {password && (
        <p className="mt-8 text-center text-green-600 dark:text-green-400 text-xl">
          Password: {password}
        </p>
      )}
      {!password && revealed && (
        <p className="mt-8 text-center text-blue-600 dark:text-blue-400 text-xl">
          Password revealed! Check the console or your agent's observations.
        </p>
      )}
    </div>
  );
};

export default RightClickRevealHard;
