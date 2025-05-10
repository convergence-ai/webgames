import { useEffect, useRef, useState } from "react";

export const PASSWORD_VerifyHard = "VERIFYHARDER2025";
export const TASK_ID_VerifyHard = "verify-challenge-hard";

export default function VerifyChallengeHard() {
  const [verificationCode, setVerificationCode] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Pre-defined email
  const email = "proxy@convergence.ai";

  // Generate a hard-to-read verification code using very similar characters
  const generateVerificationCode = () => {
    const verySimilarChars = [
      "l",
      "I",
      "1",
      "S",
      "5",
      "B",
      "8",
      "Z",
      "2",
      "Q",
      "O",
      "0",
      "D",
      "G",
      "6",
      "C",
      "U",
      "V",
      "W",
      "M",
      "N",
    ];
    let code = "";
    for (let i = 0; i < 8; i++) {
      // Longer code
      code +=
        verySimilarChars[Math.floor(Math.random() * verySimilarChars.length)];
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

    // Add very dense noisy background
    for (let i = 0; i < 300; i++) {
      // More noise
      ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.2})`;
      ctx.fillRect(
        Math.random() * canvas.width,
        Math.random() * canvas.height,
        Math.random() * 4, // Larger noise particles
        Math.random() * 4
      );
    }

    const spacing = 18; // Tighter spacing
    const totalWidth = spacing * (correctCode.length - 1);
    const startX = (canvas.width - totalWidth) / 2;

    // Draw heavily distorted text
    correctCode.split("").forEach((char, index) => {
      ctx.save();

      const x = startX + index * spacing;
      const y = 35 + (Math.random() - 0.5) * 15; // More vertical variation
      ctx.translate(x, y);
      ctx.rotate((Math.random() - 0.5) * 0.8); // More rotation

      let fontSize = Math.floor(Math.random() * 10) + 25; // Variable font size
      const fonts = [
        "Arial",
        "Times New Roman",
        "Courier New",
        "Georgia",
        "Verdana",
      ];
      const font = fonts[Math.floor(Math.random() * fonts.length)];
      ctx.font = `${Math.random() > 0.3 ? "bold" : ""} ${fontSize}px ${font}`;

      // Aggressive stretching and squashing
      ctx.scale(
        1 + (Math.random() - 0.5) * 0.6,
        1 + (Math.random() - 0.5) * 0.6
      );

      // Multiple thick strikethrough lines
      for (let i = 0; i < 3; i++) {
        if (Math.random() > 0.3) {
          ctx.strokeStyle = `rgba(0,0,0,${Math.random() * 0.4})`;
          ctx.lineWidth = Math.random() * 2.5;
          ctx.beginPath();
          ctx.moveTo(-20, Math.random() * 40 - 10);
          ctx.lineTo(30, Math.random() * 40 - 10);
          ctx.stroke();
        }
      }

      // Draw character shadow for depth
      if (Math.random() > 0.4) {
        ctx.fillStyle = "rgba(0,0,0,0.25)";
        ctx.fillText(char, Math.random() * 2 - 1, Math.random() * 2 - 1);
      }

      // Draw the main character
      ctx.fillStyle = "black";
      ctx.fillText(char, 0, 0);

      // Add overlapping effect by drawing part of the character again with offset
      if (Math.random() > 0.5) {
        ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.4 + 0.1})`;
        ctx.fillText(
          char,
          (Math.random() - 0.5) * 5,
          (Math.random() - 0.5) * 5
        );
      }

      ctx.restore();
    });

    // Add more connecting/distracting lines
    ctx.beginPath();
    for (let i = 0; i < correctCode.length - 1; i++) {
      if (Math.random() > 0.3) {
        const x1 = startX + i * spacing + (Math.random() * 10 - 5);
        const x2 = x1 + spacing + (Math.random() * 10 - 5);
        const y1 = 35 + (Math.random() - 0.5) * 15;
        const y2 = 35 + (Math.random() - 0.5) * 15;
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
    }
    ctx.strokeStyle = `rgba(0,0,0,${Math.random() * 0.2 + 0.1})`;
    ctx.lineWidth = Math.random() * 1.5;
    ctx.stroke();

    // Add more crossing lines with varying thickness and curves
    for (let i = 0; i < 6; i++) {
      ctx.beginPath();
      ctx.strokeStyle = `rgba(0,0,0,${Math.random() * 0.35})`;
      ctx.lineWidth = Math.random() * 2.5;

      const sx = Math.random() * canvas.width;
      const sy = Math.random() * canvas.height;
      const ex = Math.random() * canvas.width;
      const ey = Math.random() * canvas.height;
      const cp1x = Math.random() * canvas.width;
      const cp1y = Math.random() * canvas.height;
      const cp2x = Math.random() * canvas.width;
      const cp2y = Math.random() * canvas.height;

      ctx.moveTo(sx, sy);
      if (Math.random() > 0.5) {
        ctx.quadraticCurveTo(cp1x, cp1y, ex, ey);
      } else {
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, ex, ey);
      }
      ctx.stroke();
    }

    // Add more connecting/distracting dots
    for (let i = 0; i < 50; i++) {
      ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.5})`;
      ctx.beginPath();
      ctx.arc(
        startX + Math.random() * totalWidth * 1.2 - totalWidth * 0.1, // spread dots wider
        35 + (Math.random() - 0.5) * 30, // more vertical spread
        Math.random() * 2, // larger dots
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }, [correctCode]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (verificationCode !== correctCode) {
      setError("Incorrect verification code. Please try again.");
      setSuccess(false);
      return;
    }

    setError("");
    setSuccess(true);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      {" "}
      {/* Slightly different bg for hard mode */}
      <div className="max-w-lg w-full space-y-8">
        {" "}
        {/* Wider container */}
        <div>
          <h2 className="mt-6 text-center text-4xl font-bold text-gray-800">
            {" "}
            {/* Bolder title */}
            Advanced Verification Protocol
          </h2>
          <p className="mt-2 text-center text-md text-gray-700">
            {" "}
            {/* More formal text */}
            Enter the 8-character security token displayed below.
          </p>
        </div>
        <div className="mt-8 bg-white py-10 px-6 shadow-xl sm:rounded-lg sm:px-12">
          {" "}
          {/* More padding and shadow */}
          <div className="mb-6 p-4 bg-gray-200 rounded-lg flex justify-center">
            {" "}
            {/* Darker bg for canvas */}
            <canvas
              ref={canvasRef}
              width="280" // Wider canvas for more characters
              height="70" // Taller canvas for more distortion
              className="border-2 border-gray-300"
            />
          </div>
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-semibold text-gray-700"
              >
                {" "}
                {/* Semibold label */}
                Registered Email
              </label>
              <div className="mt-1">
                <input
                  type="text"
                  value={email}
                  disabled
                  className="bg-gray-100 w-full px-4 py-2.5 border border-gray-300 rounded-md text-gray-500 sm:text-sm focus:ring-0 focus:border-gray-400" // Custom styling
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="verification-code"
                className="block text-sm font-semibold text-gray-700"
              >
                Security Token
              </label>
              <div className="mt-1">
                <input
                  id="verification-code"
                  name="verification-code"
                  type="text"
                  required
                  value={verificationCode}
                  onChange={(e) => setVerificationCode(e.target.value)}
                  className="appearance-none block w-full px-4 py-2.5 border border-gray-400 rounded-md shadow-sm placeholder-gray-500 focus:outline-none focus:ring-red-500 focus:border-red-500 sm:text-sm" // Red focus for intensity
                  maxLength={8} // Max length for input
                />
              </div>
            </div>

            {error && (
              <div className="text-red-700 text-sm font-semibold p-2 bg-red-100 border border-red-300 rounded-md">
                {" "}
                {/* More prominent error */}
                {error}
              </div>
            )}

            {success && (
              <div className="text-green-700 text-sm font-semibold p-2 bg-green-100 border border-green-300 rounded-md">
                {" "}
                {/* More prominent success */}
                Access Granted. Password: {PASSWORD_VerifyHard}
              </div>
            )}

            <div>
              <button
                type="submit"
                className="w-full flex justify-center py-3 px-6 border border-transparent rounded-md shadow-lg text-md font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition duration-150 ease-in-out" // More prominent button
              >
                Authorize Access
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
