import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_TextMirrorHard = "SuperComplexTextMaster2024";
export const TASK_ID_TextMirrorHard = "text-mirror-hard";

const sampleTextHard = `// This is a JavaScript code snippet.
function greet(name: string): string {
  const message = \`Hello, \${name}!
  Welcome to the "Hard Text Mirror" challenge.
  Pay attention to:
    - Indentation (spaces vs. tabs - use spaces here!)
    - Special characters: \` ~ ! @ # $ % ^ & * ( ) _ + - = { } [ ] | \\\\ : ; " ' < > , . ? /
    - Line endings (LF vs. CRLF - ensure LF)
  \`;
  console.log(message.length);
  return message.trim() + \`\\nFinal line with a tab:\\tEND\`;
}

/*
  Multi-line comment example.
  The next line is intentionally blank.

  Good luck!
*/
`;

const TextMirrorHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_TextMirrorHard);
  const [userInput, setUserInput] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [password] = useState(PASSWORD_TextMirrorHard);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // For hard version, we require an exact match, including leading/trailing whitespace of the sample.
    if (userInput === sampleTextHard) {
      setIsComplete(true);
      recordSuccess();
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Text Mirror Challenge (Hard)</h1>
      <p className="mb-4">
        Copy the text exactly as shown on the left into the text area on the
        right. Every single character, including whitespace and line breaks,
        matters!
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Original Text */}
        <div className="p-4 bg-gray-100 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">Original Text:</h2>
          <div className="whitespace-pre-wrap font-mono text-sm">
            {sampleTextHard}
          </div>
        </div>

        {/* Input Area */}
        <div>
          <form onSubmit={handleSubmit}>
            <textarea
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              className="w-full h-96 p-4 border rounded-lg font-mono text-sm"
              placeholder="Type the text here..."
              spellCheck="false"
              autoCapitalize="none"
              autoCorrect="off"
            />
            <button
              type="submit"
              className="mt-4 px-6 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Submit
            </button>
          </form>
        </div>
      </div>

      {/* Success Message */}
      {isComplete && (
        <div className="mt-8 p-4 bg-green-100 text-green-700 rounded-lg">
          <h2 className="text-lg font-bold">🎉 Phenomenal!</h2>
          <p>You\'ve conquered the Hard Text Mirror challenge!</p>
          <p className="mt-2">
            The secret password is: <strong>{password}</strong>
          </p>
        </div>
      )}
    </div>
  );
};

export default TextMirrorHard;
