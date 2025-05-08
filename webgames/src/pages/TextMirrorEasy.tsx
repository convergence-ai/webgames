import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_TextMirrorEasy = "EasyPeasyText2024";
export const TASK_ID_TextMirrorEasy = "text-mirror-easy";

const sampleTextEasy = `Hello World!
This is a simple test.
Have fun!`;

const TextMirrorEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_TextMirrorEasy);
  const [userInput, setUserInput] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [password] = useState(PASSWORD_TextMirrorEasy);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (userInput.trim() === sampleTextEasy.trim()) {
      setIsComplete(true);
      recordSuccess();
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Text Mirror Challenge (Easy)</h1>
      <p className="mb-4">
        Copy the text exactly as shown on the left into the text area on the
        right. Every character matters!
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Original Text */}
        <div className="p-4 bg-gray-100 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">Original Text:</h2>
          <div className="whitespace-pre-wrap font-mono">{sampleTextEasy}</div>
        </div>

        {/* Input Area */}
        <div>
          <form onSubmit={handleSubmit}>
            <textarea
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              className="w-full h-64 p-4 border rounded-lg font-mono"
              placeholder="Type the text here..."
            />
            <button
              type="submit"
              className="mt-4 px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Submit
            </button>
          </form>
        </div>
      </div>

      {/* Success Message */}
      {isComplete && (
        <div className="mt-8 p-4 bg-green-100 text-green-700 rounded-lg">
          <h2 className="text-lg font-bold">🎉 Congratulations!</h2>
          <p>You've successfully completed the Text Mirror challenge (Easy).</p>
          <p className="mt-2">
            The secret password is: <strong>{password}</strong>
          </p>
        </div>
      )}
    </div>
  );
};

export default TextMirrorEasy;
