import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_StockMarketScrollEasy = "EASY_GOOG";
export const TASK_ID_StockMarketScrollEasy = "stock-market-easy";

const StockMarketScrollEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_StockMarketScrollEasy);
  const [answer, setAnswer] = useState("");
  const [isCorrect, setIsCorrect] = useState(false);

  const checkAnswer = () => {
    if (answer.toLowerCase() === "GOOG".toLowerCase()) {
      setIsCorrect(true);
      recordSuccess();
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="mb-8 bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold mb-4">
          Stock Market Research Challenge (Easy)
        </h1>
        <p className="text-gray-600 mb-6">
          Read the short report below and find the recommended stock.
        </p>

        <div className="flex gap-4">
          <input
            type="text"
            placeholder="Enter stock ticker"
            className="flex-1 p-2 border rounded"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && checkAnswer()}
          />
          <button
            onClick={checkAnswer}
            className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
          >
            Submit
          </button>
        </div>

        {isCorrect && (
          <div className="mt-4 p-4 bg-green-100 text-green-700 rounded">
            Correct! The password is:{" "}
            <span className="font-bold">{PASSWORD_StockMarketScrollEasy}</span>
          </div>
        )}
      </div>

      <div className="bg-gray-50 rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Top Tech Pick for 2025</h2>
        <p className="text-gray-800">
          After careful analysis, our top recommendation for a tech stock in
          2025 is Google (Alphabet). Their innovations in AI make them a strong
          buy. The stock ticker is:{" "}
          <span className="font-bold text-xl">GOOG</span>.
        </p>
      </div>
    </div>
  );
};

export default StockMarketScrollEasy;
