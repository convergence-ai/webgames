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

export const TASK_ID_ChartReadHard = "chart-read-hard";
export const PASSWORD_ChartReadHard = "HARD_CHART_EXPERT_2024";

const ChartReadHard: FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ChartReadHard);
  const [data, setData] = useState<DataPoint[]>([]);
  const [maxPrice, setMaxPrice] = useState("");
  const [maxTime, setMaxTime] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const generateData = () => {
      const points: DataPoint[] = [];
      const maxPoints = 48; // More data points (e.g., every 30 minutes)

      // Introduce multiple potential peaks or a less obvious main peak
      const numberOfPeaks = 2 + Math.floor(Math.random() * 2); // 2 or 3 potential peaks
      const peakHours: number[] = [];
      for (let p = 0; p < numberOfPeaks; p++) {
        peakHours.push(Math.floor(Math.random() * (maxPoints - 4)) + 2); // Avoid edges for peaks
      }

      const trueSpikeIndex =
        peakHours[Math.floor(Math.random() * peakHours.length)];
      const trueSpikeMagnitude = 260 + Math.random() * 40; // Main spike higher

      for (let i = 0; i < maxPoints; i++) {
        let price;
        const isNearTrueSpike = Math.abs(i - trueSpikeIndex) <= 1;
        const isNearOtherPeak = peakHours.some(
          (ph) => ph !== trueSpikeIndex && Math.abs(i - ph) <= 1
        );

        if (i === trueSpikeIndex) {
          price = trueSpikeMagnitude;
        } else if (isNearTrueSpike) {
          price =
            trueSpikeMagnitude -
            (10 + Math.random() * 10) * Math.abs(i - trueSpikeIndex) -
            Math.random() * 10; // Steeper dropoff
        } else if (isNearOtherPeak) {
          const otherPeakHour =
            peakHours.find(
              (ph) => ph !== trueSpikeIndex && Math.abs(i - ph) <= 1
            ) || 0;
          const otherPeakMagnitude = 230 + Math.random() * 20; // Other peaks are lower
          if (i === otherPeakHour) {
            price = otherPeakMagnitude;
          } else {
            price =
              otherPeakMagnitude -
              (15 + Math.random() * 5) * Math.abs(i - otherPeakHour) -
              Math.random() * 5;
          }
        } else {
          // More volatile base price
          price =
            120 +
            Math.sin(i * 0.3 + Math.random()) * 30 +
            (Math.random() - 0.7) * 40;
        }

        price = Math.max(50, Math.min(300, price));
        const minute = (i * 30) % 60;
        const hour = Math.floor((i * 30) / 60);
        points.push({
          time: `${String(hour).padStart(2, "0")}:${String(minute).padStart(
            2,
            "0"
          )}`,
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
    // Stricter tolerance for hard version: 2.5%
    const priceThreshold = actualMax.price * 0.025;

    if (
      Math.abs(submittedPrice - actualMax.price) <= priceThreshold &&
      maxTime === actualMax.time
    ) {
      setIsComplete(true);
      recordSuccess();
      setError("");
    } else {
      let errorMessage = "Incorrect. ";
      if (maxTime !== actualMax.time) {
        errorMessage += `Time is wrong. `;
      }
      if (Math.abs(submittedPrice - actualMax.price) > priceThreshold) {
        const diff = submittedPrice - actualMax.price;
        errorMessage += `Price is off by ${diff > 0 ? "+" : ""}${diff.toFixed(
          2
        )}. `;
      }
      setError(errorMessage + "Try again!");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-500 via-pink-600 to-purple-700 p-8">
      <div className="flex justify-center mb-8">
        <div className="bg-white rounded-xl p-6 max-w-2xl shadow-sm">
          <h1 className="text-3xl font-bold text-center mb-4 text-gray-800">
            Chart Read Challenge (Hard)
          </h1>
          <p className="text-center text-gray-600">
            Identify the absolute maximum price and the precise time it
            occurred. The price must be within 2.5% of the actual maximum. The
            chart is more volatile.
          </p>
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 mb-8 shadow-lg">
        <div className="h-[450px]">
          {" "}
          {/* Slightly taller chart for more data */}
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10 }}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={["auto", "auto"]}
                tickFormatter={(value) => `$${value.toFixed(0)}`}
              />
              <Tooltip formatter={(value: number) => value.toFixed(2)} />
              <Line
                type="monotone"
                dataKey="price"
                stroke="#ef4444" // Red line for hard
                strokeWidth={1.5} // Thinner line
                dot={false} // No dots for a cleaner, harder to read look
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
                <label
                  htmlFor="maxPriceHard"
                  className="block text-sm font-medium text-gray-700"
                >
                  Maximum Price
                </label>
                <input
                  id="maxPriceHard"
                  type="number"
                  step="0.01"
                  value={maxPrice}
                  onChange={(e) => setMaxPrice(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-red-500 focus:ring-red-500"
                  placeholder="Enter precise peak price..."
                  required
                />
              </div>
              <div>
                <label
                  htmlFor="maxTimeHard"
                  className="block text-sm font-medium text-gray-700"
                >
                  Time of Maximum (HH:MM)
                </label>
                {/* For hard version, consider a text input if times are too many for a select */}
                {/* Or a more complex select if needed, for now, it's dynamic based on generated data */}
                <select
                  id="maxTimeHard"
                  value={maxTime}
                  onChange={(e) => setMaxTime(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-red-500 focus:ring-red-500"
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
                className="w-full bg-red-600 text-white py-2 px-4 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
              >
                Submit Answer
              </button>
              {error && (
                <p className="text-yellow-500 bg-yellow-100 border border-yellow-300 rounded p-2 text-xs text-center">
                  {error}
                </p>
              )}
            </form>
          ) : (
            <div className="text-center">
              <h2 className="text-2xl font-bold text-purple-700 mb-2">
                Challenge Mastered!
              </h2>
              <p className="text-gray-700">
                The password is: {PASSWORD_ChartReadHard}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChartReadHard;
