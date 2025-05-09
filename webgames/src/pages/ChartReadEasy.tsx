import React, { FC, useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface DataPoint {
  time: string;
  price: number;
}

export const TASK_ID_ChartReadEasy = "chart-read-easy";
export const PASSWORD_ChartReadEasy = "EASY_CHART_PEAK_2024";

const ChartReadEasy: FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ChartReadEasy);
  const [data, setData] = useState<DataPoint[]>([]);
  const [maxPrice, setMaxPrice] = useState("");
  const [maxTime, setMaxTime] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState("");

  // Generate simplified stock data
  useEffect(() => {
    const generateData = () => {
      const points: DataPoint[] = [];
      const maxPoints = 12; // Fewer data points (e.g., every 2 hours)
      const spikeHourIndex = 3 + Math.floor(Math.random() * 6); // Spike between 06:00 and 16:00

      for (let i = 0; i < maxPoints; i++) {
        let price;
        const hour = i * 2; // Time in 2-hour increments

        if (i === spikeHourIndex) {
          // The spike point - make it more obvious
          price = 200 + Math.random() * 20; // Higher base for spike
        } else {
          // Other points - less fluctuation
          price = 80 + Math.sin(i * 0.8) * 10 + (Math.random() - 0.5) * 10;
        }
        price = Math.max(60, Math.min(220, price)); // Adjusted bounds
        points.push({
          time: `${String(hour).padStart(2, "0")}:00`,
          price: Math.round(price * 100) / 100,
        });
      }
      return points;
    };

    const newData = generateData();
    setData(newData);
  }, []);

  const actualMax = data.reduce(
    (max, point) => (point.price > max.price ? point : max),
    { time: "", price: -Infinity }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const submittedPrice = parseFloat(maxPrice);
    const priceThreshold = actualMax.price * 0.1; // 10% tolerance for easy version

    if (
      Math.abs(submittedPrice - actualMax.price) <= priceThreshold &&
      maxTime === actualMax.time
    ) {
      setIsComplete(true);
      recordSuccess();
      setError("");
    } else {
      setError(
        `Incorrect. Expected around ${actualMax.price.toFixed(2)} at ${
          actualMax.time
        }. Try again!`
      );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-400 via-cyan-400 to-blue-400 p-8">
      <div className="flex justify-center mb-8">
        <div className="bg-white rounded-xl p-6 max-w-2xl shadow-sm">
          <h1 className="text-3xl font-bold text-center mb-4 text-gray-800">
            Chart Read Challenge (Easy)
          </h1>
          <p className="text-center text-gray-600">
            Find the highest price and the time it occurred. The price needs to
            be within 10% of the actual maximum.
          </p>
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 mb-8 shadow-lg">
        <div className="h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={["auto", "auto"]} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="price"
                stroke="#4ade80" // Green line for easy
                strokeWidth={2}
                dot={{ r: 4 }} // Make points more visible
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="flex justify-center">
        <div className="bg-white rounded-xl p-6 w-full max-w-md shadow-lg">
          {!isComplete ? (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Maximum Price
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={maxPrice}
                  onChange={(e) => setMaxPrice(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
                  placeholder="Enter peak price..."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Time of Maximum (HH:00)
                </label>
                <select
                  value={maxTime}
                  onChange={(e) => setMaxTime(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
                  required
                >
                  <option value="">Select time...</option>
                  {data.map((point) => (
                    <option key={point.time} value={point.time}>
                      {point.time}
                    </option>
                  ))}
                </select>
              </div>
              <button
                type="submit"
                className="w-full bg-green-500 text-white py-2 px-4 rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
              >
                Submit
              </button>
              {error && (
                <p className="text-red-600 text-sm text-center">{error}</p>
              )}
            </form>
          ) : (
            <div className="text-center">
              <h2 className="text-2xl font-bold text-green-700 mb-2">
                Well Done!
              </h2>
              <p className="text-gray-700">
                The password is: {PASSWORD_ChartReadEasy}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChartReadEasy;
