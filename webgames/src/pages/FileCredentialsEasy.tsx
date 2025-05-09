import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_FileCredentialsEasy = "EasyFileSecrets2024";
export const TASK_ID_FileCredentialsEasy = "file-credentials-easy";

const FileCredentialsEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_FileCredentialsEasy);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<{
    text: string;
    type: "success" | "error" | null;
  }>({ text: "", type: null });
  const [isComplete, setIsComplete] = useState(false);

  const fixedUsername = "EasyUser";
  const fixedPassword = "easy";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (username === fixedUsername && password === fixedPassword) {
      setIsComplete(true);
      recordSuccess();
      setMessage({
        text: `Login successful! The completion password is: ${PASSWORD_FileCredentialsEasy}`,
        type: "success",
      });
    } else {
      setMessage({
        text: "Invalid credentials. Please use the credentials shown on the page.",
        type: "error",
      });
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-green-100 to-blue-100 p-4">
      <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold text-center mb-6 text-green-600">
          File Credentials (Easy)
        </h1>

        <div className="mb-6 text-center p-4 bg-gray-100 rounded-lg">
          <p className="text-lg font-semibold text-gray-700">
            Use these credentials to log in:
          </p>
          <p className="text-md text-gray-600 mt-1">
            Username:{" "}
            <span className="font-mono text-green-700">{fixedUsername}</span>
          </p>
          <p className="text-md text-gray-600">
            Password:{" "}
            <span className="font-mono text-green-700">{fixedPassword}</span>
          </p>
        </div>

        {message.text && (
          <div
            className={`mb-4 p-3 rounded-lg text-center ${
              message.type === "error"
                ? "bg-red-100 text-red-700"
                : message.type === "success"
                ? "bg-green-100 text-green-700"
                : ""
            }`}
          >
            {message.text}
          </div>
        )}

        {!isComplete && (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition-colors"
            >
              Login
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default FileCredentialsEasy;
