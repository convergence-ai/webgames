import { useEffect, useRef, useState } from "react";

export const PASSWORD_VerifyEasy = "VERIFYEASY2025";
export const TASK_ID_VerifyEasy = "verify-challenge-easy";

export default function VerifyChallengeEasy() {
  const [verificationCode, setVerificationCode] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Pre-defined email
  const email = "proxy@convergence.ai";

  // Generate an easy-to-read verification code
  const generateVerificationCode = () => {
    const distinctChars = [
      "A",
      "B",
      "C",
      "E",
      "F",
      "H",
      "K",
      "M",
      "N",
      "P",
      "R",
      "T",
      "W",
      "X",
      "Y",
      "3",
      "4",
      "7",
      "8",
    ];
    let code = "";
    for (let i = 0; i < 4; i++) {
      // Shorter code
      code += distinctChars[Math.floor(Math.random() * distinctChars.length)];
    }
    return code;
  };

  const [correctCode] = useState(generateVerificationCode());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Add light noisy background
    for (let i = 0; i < 50; i++) {
      // Less noise
      ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.1})`;
      ctx.fillRect(
        Math.random() * canvas.width,
        Math.random() * canvas.height,
        Math.random() * 2,
        Math.random() * 2
      );
    }

    const spacing = 25; // More spacing for clarity
    const totalWidth = spacing * (correctCode.length - 1);
    const startX = (canvas.width - totalWidth) / 2;

    // Draw simple text
    correctCode.split("").forEach((char, index) => {
      ctx.save();

      const x = startX + index * spacing;
      const y = 35 + (Math.random() - 0.5) * 4; // Less vertical variation
      ctx.translate(x, y);
      ctx.rotate((Math.random() - 0.5) * 0.1); // Less rotation

      const fontSize = 30; // Consistent font size
      const font = "Arial"; // Clear font
      ctx.font = `bold ${fontSize}px ${font}`;

      ctx.fillStyle = "black";
      ctx.fillText(char, 0, 0);

      ctx.restore();
    });
  }, [correctCode]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (verificationCode !== correctCode) {
      setError("Incorrect verification code");
      setSuccess(false);
      return;
    }

    setError("");
    setSuccess(true);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Verification Challenge (Easy)
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Enter the 4-digit verification code below
          </p>
        </div>

        <div className="mt-8 bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <div className="mb-4 p-4 bg-gray-100 rounded-md flex justify-center">
            <canvas
              ref={canvasRef}
              width="150" // Adjusted for shorter code
              height="60"
              className="border border-gray-200"
            />
          </div>

          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700"
              >
                Email address
              </label>
              <div className="mt-1">
                <input
                  type="text"
                  value={email}
                  disabled
                  className="bg-gray-50 w-full px-3 py-2 border border-gray-300 rounded-md text-gray-500 sm:text-sm"
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="verification-code"
                className="block text-sm font-medium text-gray-700"
              >
                Verification Code
              </label>
              <div className="mt-1">
                <input
                  id="verification-code"
                  name="verification-code"
                  type="text"
                  required
                  value={verificationCode}
                  onChange={(e) => setVerificationCode(e.target.value)}
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  maxLength={4} // Max length for input
                />
              </div>
            </div>

            {error && (
              <div className="text-red-600 text-sm font-medium">{error}</div>
            )}

            {success && (
              <div className="text-emerald-600 text-sm font-medium">
                Success! The password is: {PASSWORD_VerifyEasy}
              </div>
            )}

            <div>
              <button
                type="submit"
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Verify
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
