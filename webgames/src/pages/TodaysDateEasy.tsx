import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_TodaysDateEasy = "DATE_EASY_2024";
export const TASK_ID_TodaysDateEasy = "dateeasy";

const TodaysDateEasy: React.FC = () => {
  const [date, setDate] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const { recordSuccess } = useTaskAnalytics(TASK_ID_TodaysDateEasy);

  const handleDateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const today = new Date().toISOString().split("T")[0];
    if (date === today) {
      setShowPassword(true);
      await recordSuccess();
    } else {
      setShowPassword(false);
    }
  };

  const todayStr = new Date().toISOString().split("T")[0];

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-2xl font-bold mb-6">Today's Date Task (Easy)</h1>
      <form onSubmit={handleDateSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="date"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Enter today's date in UTC (Format: YYYY-MM-DD)
          </label>
          <input
            type="text"
            id="date"
            placeholder="1993-11-29"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <button
          type="button"
          onClick={() => setShowHint(true)}
          className="w-full bg-gray-300 text-gray-700 py-2 px-4 rounded hover:bg-gray-400 transition-colors"
        >
          Show Hint
        </button>
        {showHint && (
          <div className="p-2 bg-yellow-100 rounded">
            <p className="text-yellow-800 text-sm">
              Hint: Today's date is {todayStr}
            </p>
          </div>
        )}
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition-colors"
        >
          Check Date
        </button>
      </form>

      {showPassword && (
        <div className="mt-6 p-4 bg-green-100 rounded">
          <p className="text-green-800 font-medium">
            Secret Password: {PASSWORD_TodaysDateEasy}
          </p>
        </div>
      )}
    </div>
  );
};

export default TodaysDateEasy;
