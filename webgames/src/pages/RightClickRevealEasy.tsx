import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_RightClickRevealEasy = "EASY_RIGHT_CLICK";
export const TASK_ID_RightClickRevealEasy = "right-click-easy";

const RightClickRevealEasy = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_RightClickRevealEasy);
  const [password, setPassword] = useState("");

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setPassword(PASSWORD_RightClickRevealEasy);
    recordSuccess();
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Right Click Challenge (Easy)</h1>

      <p className="mb-6">
        Simply right-click anywhere in the large box below to see the password.
        This one is super easy!
      </p>

      <div
        className="p-12 text-center bg-green-100 dark:bg-green-900 rounded-lg shadow-xl cursor-context-menu hover:bg-green-200 dark:hover:bg-green-800 transition-colors"
        onContextMenu={handleContextMenu}
        style={{ minHeight: "200px" }}
      >
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
          Right Click Here!
        </h2>
        {password && (
          <p className="mt-6 text-xl text-green-700 dark:text-green-300">
            Password: {password}
          </p>
        )}
      </div>
    </div>
  );
};

export default RightClickRevealEasy;
