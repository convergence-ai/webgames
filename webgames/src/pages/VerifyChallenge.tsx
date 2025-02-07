import { useState, useEffect, useRef } from 'react';

export const PASSWORD_Verify = "VERIFYMASTER2025";
export const TASK_ID_Verify = "verify-challenge";

export default function VerifyChallenge() {
  const [verificationCode, setVerificationCode] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Pre-defined email
  const email = "proxy@convergence.ai";

  // Generate a hard-to-read verification code using distinct characters
  const generateVerificationCode = () => {
    // Added back '0' and 'O' with specific font handling later
    const similarChars = ['L', 'I', 'S', 'B', 'Z', 'Q', 'G', 'C', 'D', 'W', 'M', 'N', '0', 'O'];
    let code = '';
    for (let i = 0; i < 6; i++) {
      code += similarChars[Math.floor(Math.random() * similarChars.length)];
    }
    return code;
  };

  const [correctCode] = useState(generateVerificationCode());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Add noisy background with more density
    for (let i = 0; i < 200; i++) {
      ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.15})`;
      ctx.fillRect(
        Math.random() * canvas.width,
        Math.random() * canvas.height,
        Math.random() * 3,
        Math.random() * 3
      );
    }

    // Calculate total width to center the text
    const spacing = 20; // Reduced spacing between characters
    const totalWidth = spacing * (correctCode.length - 1);
    const startX = (canvas.width - totalWidth) / 2;

    // Draw distorted text with more effects
    correctCode.split('').forEach((char, index) => {
      ctx.save();
      
      // Closer spacing and more vertical variation
      const x = startX + index * spacing;
      const y = 35 + (Math.random() - 0.5) * 12;
      ctx.translate(x, y);
      ctx.rotate((Math.random() - 0.5) * 0.6);
      
      // Special handling for '0' and 'O' to make them distinct
      let fontSize = Math.floor(Math.random() * 8) + 28; // Slightly larger base size
      let font = '';
      
      if (char === '0') {
        font = 'Courier New';
        ctx.font = `bold ${fontSize}px ${font}`;
        // Add a slash through zero
        ctx.strokeStyle = 'rgba(0,0,0,0.8)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(-5, -fontSize/2);
        ctx.lineTo(5, fontSize/2);
        ctx.stroke();
      } else if (char === 'O') {
        font = 'Georgia';
        fontSize += 2;
        ctx.font = `bold ${fontSize}px ${font}`;
      } else {
        const fonts = ['Times New Roman', 'Georgia', 'Courier New'];
        font = fonts[Math.floor(Math.random() * fonts.length)];
        ctx.font = `${Math.random() > 0.5 ? 'bold' : ''} ${fontSize}px ${font}`;
      }
      
      // More aggressive stretching for non-0/O characters
      if (char !== '0' && char !== 'O') {
        ctx.scale(1 + (Math.random() - 0.5) * 0.4, 1 + (Math.random() - 0.5) * 0.4);
      }

      // Add multiple strikethrough lines (but not for 0)
      if (char !== '0') {
        for (let i = 0; i < 2; i++) {
          if (Math.random() > 0.4) {
            ctx.strokeStyle = `rgba(0,0,0,${Math.random() * 0.3})`;
            ctx.lineWidth = Math.random() * 1.5;
            ctx.beginPath();
            ctx.moveTo(-15, Math.random() * 40);
            ctx.lineTo(25, Math.random() * 40);
            ctx.stroke();
          }
        }
      }

      // Draw character shadow for depth
      if (Math.random() > 0.5 && char !== '0') {
        ctx.fillStyle = 'rgba(0,0,0,0.2)';
        ctx.fillText(char, 1, 1);
      }

      // Draw the main character
      ctx.fillStyle = 'black';
      ctx.fillText(char, 0, 0);
      
      // Add overlapping effect by drawing part of the character again
      if (Math.random() > 0.6 && char !== '0' && char !== 'O') {
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fillText(char, -2, 0);
      }
      
      ctx.restore();
    });

    // Add connecting lines between characters
    ctx.beginPath();
    for (let i = 0; i < correctCode.length - 1; i++) {
      if (Math.random() > 0.5) {
        const x1 = startX + i * spacing + 15;
        const x2 = x1 + spacing - 10;
        const y = 35 + (Math.random() - 0.5) * 10;
        ctx.moveTo(x1, y);
        ctx.lineTo(x2, y + (Math.random() - 0.5) * 8);
      }
    }
    ctx.strokeStyle = 'rgba(0,0,0,0.1)';
    ctx.stroke();

    // Add crossing lines with varying thickness
    for (let i = 0; i < 4; i++) {
      ctx.beginPath();
      ctx.strokeStyle = `rgba(0,0,0,${Math.random() * 0.3})`;
      ctx.lineWidth = Math.random() * 2;
      
      // Curved lines instead of straight
      const startX = Math.random() * canvas.width;
      const startY = 0;
      const endX = Math.random() * canvas.width;
      const endY = canvas.height;
      const controlX = Math.random() * canvas.width;
      const controlY = Math.random() * canvas.height;
      
      ctx.moveTo(startX, startY);
      ctx.quadraticCurveTo(controlX, controlY, endX, endY);
      ctx.stroke();
    }

    // Add connecting dots between characters
    for (let i = 0; i < 30; i++) {
      ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.4})`;
      ctx.beginPath();
      ctx.arc(
        startX + Math.random() * totalWidth,
        35 + (Math.random() - 0.5) * 20,
        Math.random() * 1.5,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }, [correctCode]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (verificationCode !== correctCode) {
      setError('Incorrect verification code');
      setSuccess(false);
      return;
    }

    setError('');
    setSuccess(true);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Verification Challenge
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Enter the verification code below
          </p>
        </div>

        <div className="mt-8 bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <div className="mb-4 p-4 bg-gray-100 rounded-md flex justify-center">
            <canvas 
              ref={canvasRef}
              width="200"
              height="60"
              className="border border-gray-200"
            />
          </div>

          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
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
              <label htmlFor="verification-code" className="block text-sm font-medium text-gray-700">
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
                />
              </div>
            </div>

            {error && (
              <div className="text-red-600 text-sm font-medium">
                {error}
              </div>
            )}

            {success && (
              <div className="text-emerald-600 text-sm font-medium">
                Success! The password is: {PASSWORD_Verify}
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