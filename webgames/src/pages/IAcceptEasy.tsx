import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_IAcceptEasy = "i-accept-easy";
export const PASSWORD_IAcceptEasy = "EasyToAgree";

const IAcceptEasy = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_IAcceptEasy);
  const [isChecked, setIsChecked] = useState(true); // Pre-checked
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isChecked) {
      setShowPassword(true);
      recordSuccess();
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Quick Human Verification</h2>

        <div className="mb-6">
          <p className="text-gray-700 mb-4">Please confirm you are human.</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="terms"
              checked={isChecked}
              onChange={(e) => setIsChecked(e.target.checked)}
              className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
            />
            <label htmlFor="terms" className="ml-2 text-gray-700">
              I am human
            </label>
          </div>

          <button
            type="submit"
            disabled={!isChecked}
            className={`w-full py-2 px-4 rounded-md text-white font-medium
              ${
                isChecked
                  ? "bg-blue-600 hover:bg-blue-700"
                  : "bg-gray-400 cursor-not-allowed"
              }`}
          >
            Submit
          </button>
        </form>

        {showPassword && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-md">
            <p className="text-green-800 font-medium">
              Secret Password: {PASSWORD_IAcceptEasy}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default IAcceptEasy;
