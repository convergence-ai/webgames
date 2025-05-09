import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface DataPoint {
  time: string;
  primary: number;
  secondary: number;
}

function generateData(): DataPoint[] {
  const points: DataPoint[] = [];
  const hours = 2; // Reduced hours for easy version

  for (let i = 0; i < hours; i++) {
    const hour = i.toString().padStart(2, "0");
    points.push({
      time: `${hour}:00`,
      primary: Math.floor(Math.random() * 80) + 20,
      secondary: Math.floor(Math.random() * 80) + 20,
    });
  }
  return points;
}

export const TASK_ID_ChartTranscribeEasy = "chart-transcribe-easy";
export const PASSWORD_ChartTranscribeEasy = "DataScribeEasy2024";

const ChartTranscribeEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ChartTranscribeEasy);
  const [data, setData] = useState<DataPoint[]>([]);
  useEffect(() => {
    setData(generateData());
  }, []);
  const [userInput, setUserInput] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Split input into lines and parse each line
    const lines = userInput.trim().split("\n");
    const parsedInput = lines.map((line) => {
      const [time, primaryStr, secondaryStr] = line
        .split(",")
        .map((s) => s.trim());
      return {
        time,
        primary: parseInt(primaryStr),
        secondary: parseInt(secondaryStr),
      };
    });

    // Check if the input matches the data
    const isCorrect = parsedInput.every((input, index) => {
      const target = data[index];
      return (
        input.time === target.time &&
        input.primary === target.primary &&
        input.secondary === target.secondary
      );
    });

    if (isCorrect && parsedInput.length === data.length) {
      setIsComplete(true);
      recordSuccess();
      setError("");
    } else {
      setError(
        "Incorrect transcription. Please check your values and try again!"
      );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-400 via-teal-500 to-blue-500 p-8">
      {/* Instructions Card */}
      <div className="flex justify-center mb-8">
        <div className="bg-white rounded-xl p-6 max-w-2xl shadow-sm">
          <h1 className="text-3xl font-bold text-center mb-4 text-gray-800">
            Chart Transcribe Challenge (Easy)
          </h1>
          <p className="text-center text-gray-600">
            Study the bar chart and transcribe both series of data in CSV
            format. Each line should contain three values: time, primary value,
            secondary value (e.g., "00:00,45,32"). Include all data points to
            complete the challenge! The primary series is the purple bar, and
            the secondary series is the blue bar.
          </p>
        </div>
      </div>

      {/* Success Message */}
      {isComplete && (
        <div className="flex justify-center mb-8">
          <div className="bg-green-100 border border-green-200 rounded-lg p-6 shadow-lg w-full max-w-2xl">
            <h2 className="text-2xl font-bold text-green-800 mb-2">
              Congratulations!
            </h2>
            <p className="text-green-700">
              The password is: {PASSWORD_ChartTranscribeEasy}
            </p>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="bg-white rounded-xl p-6 mb-8 shadow-lg">
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="primary" fill="#8b5cf6" name="Primary" />
              <Bar dataKey="secondary" fill="#06b6d4" name="Secondary" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Input Form */}
      <div className="flex justify-center">
        <div className="bg-white rounded-xl p-6 w-full max-w-2xl shadow-lg">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Enter the data (one entry per line):
              </label>
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                className="w-full h-32 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" // Reduced height
                placeholder="00:00,45,32\n01:00,67,89\n..."
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Submit
            </button>
            {error && (
              <p className="text-red-500 text-sm text-center">{error}</p>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default ChartTranscribeEasy;
