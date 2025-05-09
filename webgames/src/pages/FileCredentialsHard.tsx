import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_FileCredentialsHard = "HardFileSecretsJPEG2024";
export const TASK_ID_FileCredentialsHard = "file-credentials-hard";

const FileCredentialsHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_FileCredentialsHard);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<{
    text: string;
    type: "success" | "error" | null;
  }>({ text: "", type: null });
  const [isComplete, setIsComplete] = useState(false);

  const correctUsername = "ImageUser";
  const passwordPartInImage = "PixelPassOnly";
  const downloadableFilename = "auth_details.jpg";
  const correctFullPassword = passwordPartInImage;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (username === correctUsername && password === correctFullPassword) {
      setIsComplete(true);
      recordSuccess();
      setMessage({
        text: `Login successful! The completion password is: ${PASSWORD_FileCredentialsHard}`,
        type: "success",
      });
    } else {
      let errorText = "Invalid credentials. ";
      if (username !== correctUsername) {
        errorText += "Check the username. ";
      }
      if (password !== correctFullPassword) {
        errorText +=
          "The password is the 'Code Fragment' found in the downloaded image. ";
      }
      setMessage({
        text: errorText.trim(),
        type: "error",
      });
    }
  };

  const handleDownload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 180;
    const ctx = canvas.getContext("2d");

    if (ctx) {
      ctx.fillStyle = "#EAEAEA";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = "#D32F2F";
      ctx.font = "bold 24px 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("SECRET CREDENTIALS", canvas.width / 2, 40);

      ctx.fillStyle = "#212121";
      ctx.font = "18px 'Courier New', Courier, monospace";
      ctx.textAlign = "left";
      ctx.fillText(`Agent ID: ${correctUsername}`, 30, 80);

      ctx.fillText(`Password: ${correctFullPassword}`, 30, 110);

      ctx.font = "italic 14px Arial";
      ctx.fillStyle = "#555";
      ctx.textAlign = "center";
      ctx.fillText(
        "(Use the Agent ID and Password above to log in)",
        canvas.width / 2,
        150
      );
    }

    const dataURL = canvas.toDataURL("image/jpeg");
    const a = document.createElement("a");
    a.href = dataURL;
    a.download = downloadableFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-red-200 via-orange-200 to-yellow-200 p-4">
      <div className="max-w-xl w-full bg-white rounded-xl shadow-2xl p-8 transform transition-all hover:scale-105">
        <h1 className="text-4xl font-black text-center mb-8 text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-orange-500">
          File Credentials (Hard)
        </h1>

        <div className="mb-8 text-center">
          <button
            onClick={handleDownload}
            className="bg-gradient-to-r from-red-600 to-orange-500 text-white px-8 py-4 rounded-lg hover:from-red-700 hover:to-orange-600 transition-all duration-300 ease-in-out text-lg font-semibold shadow-xl focus:outline-none focus:ring-4 focus:ring-orange-300"
          >
            Download Credentials Image (JPG)
          </button>
          <p className="text-md text-gray-700 mt-4">
            Download the JPEG. The username and password are in the image.
          </p>
        </div>

        {message.text && (
          <div
            className={`mb-6 p-4 rounded-lg text-center font-medium ${
              message.type === "error"
                ? "bg-red-200 text-red-800"
                : message.type === "success"
                ? "bg-green-200 text-green-800"
                : ""
            }`}
          >
            {message.text}
          </div>
        )}

        {!isComplete && (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-lg font-medium text-gray-800 mb-1">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="mt-1 block w-full rounded-lg border-gray-400 shadow-sm focus:border-red-500 focus:ring focus:ring-red-500 focus:ring-opacity-50 p-3 text-lg placeholder-gray-400"
                required
                placeholder="e.g., AgentID123"
              />
            </div>
            <div>
              <label className="block text-lg font-medium text-gray-800 mb-1">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 block w-full rounded-lg border-gray-400 shadow-sm focus:border-red-500 focus:ring focus:ring-red-500 focus:ring-opacity-50 p-3 text-lg placeholder-gray-400"
                required
                placeholder="Password from image"
              />
            </div>
            <button
              type="submit"
              className="w-full bg-gradient-to-r from-red-600 to-orange-600 text-white px-6 py-4 rounded-lg hover:from-red-700 hover:to-orange-700 transition-all duration-300 ease-in-out text-xl font-bold shadow-2xl focus:outline-none focus:ring-4 focus:ring-orange-300 mt-4"
            >
              Login
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default FileCredentialsHard;
