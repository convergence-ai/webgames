import React, { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_WebGLTextHard = "WEBGLSTAR2024HARD";
export const TASK_ID_WebGLTextHard = "webgl-text-hard";

// Vertex shader source (same as original)
const vertexShaderSource = `#version 300 es
precision highp float;
layout (location=0) in vec4 position;
layout (location=1) in vec3 color;

out vec3 vColor;

void main() {
    vColor = color;
    gl_Position = position;
}`;

// Fragment shader source (same as original)
const fragmentShaderSource = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 fragColor;

void main() {
    fragColor = vec4(vColor, 1.0);
}`;

// Star vertices calculation
const getStarPositions = () => {
  const points = [];
  const R = 0.5; // Outer radius
  const r = 0.22; // Inner radius
  for (let i = 0; i < 5; i++) {
    // Outer point
    points.push(
      R * Math.cos(Math.PI / 2 + (i * 2 * Math.PI) / 5),
      R * Math.sin(Math.PI / 2 + (i * 2 * Math.PI) / 5),
      0.0
    );
    // Inner point
    points.push(
      r * Math.cos(Math.PI / 2 + ((i + 0.5) * 2 * Math.PI) / 5),
      r * Math.sin(Math.PI / 2 + ((i + 0.5) * 2 * Math.PI) / 5),
      0.0
    );
  }
  // p[0]=P0, p[1]=I0, p[2]=P1, p[3]=I1, ..., p[8]=P4, p[9]=I4
  const p = [];
  for (let i = 0; i < points.length / 3; i++) {
    p.push([points[i * 3], points[i * 3 + 1], points[i * 3 + 2]]);
  }

  return new Float32Array([
    ...p[0],
    ...p[1],
    ...p[9], // P0, I0, I4
    ...p[2],
    ...p[3],
    ...p[1], // P1, I1, I0
    ...p[4],
    ...p[5],
    ...p[3], // P2, I2, I1
    ...p[6],
    ...p[7],
    ...p[5], // P3, I3, I2
    ...p[8],
    ...p[9],
    ...p[7], // P4, I4, I3
    // Inner pentagon triangles (fanned from I0)
    ...p[1],
    ...p[3],
    ...p[5], // I0, I1, I2
    ...p[1],
    ...p[5],
    ...p[7], // I0, I2, I3
    ...p[1],
    ...p[7],
    ...p[9], // I0, I3, I4
  ]);
};

const starPositions = getStarPositions();
const numStarVertices = starPositions.length / 3;

// Colors for the star (all yellow)
const starColors = new Float32Array(numStarVertices * 3);
for (let i = 0; i < numStarVertices; i++) {
  starColors[i * 3 + 0] = 1.0; // R
  starColors[i * 3 + 1] = 0.84; // G (gold-ish yellow)
  starColors[i * 3 + 2] = 0.0; // B
}

const WebGLTextHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_WebGLTextHard);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [userInput, setUserInput] = useState("");
  const [isCorrect, setIsCorrect] = useState(false);
  const [scale, setScale] = useState(1);
  const targetText = "STAR";

  useEffect(() => {
    const updateScale = () => {
      if (containerRef.current?.parentElement) {
        const parent = containerRef.current.parentElement;
        const maxWidth = parent.clientWidth;
        const maxHeight = window.innerHeight - 150;
        const scaleX = maxWidth / 800;
        const scaleY = maxHeight / 600;
        setScale(Math.min(scaleX, scaleY));
      }
    };

    updateScale();
    window.addEventListener("resize", updateScale);
    return () => window.removeEventListener("resize", updateScale);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2");
    if (!gl) {
      alert("WebGL2 not supported");
      return;
    }

    const createShader = (source: string, type: number) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = createShader(vertexShaderSource, gl.VERTEX_SHADER);
    const fragmentShader = createShader(
      fragmentShaderSource,
      gl.FRAGMENT_SHADER
    );
    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    if (!program) return;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
      return;
    }

    gl.detachShader(program, vertexShader);
    gl.detachShader(program, fragmentShader);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, starPositions, gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, starColors, gl.STATIC_DRAW);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(1);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.useProgram(program);

    let rotation = 0;
    const animate = () => {
      rotation += 0.02; // Faster rotation
      gl.clear(gl.COLOR_BUFFER_BIT);

      const rotatedPositions = Array.from(starPositions); // Create a mutable copy
      for (let i = 0; i < numStarVertices; i++) {
        const x = starPositions[i * 3];
        const y = starPositions[i * 3 + 1];
        rotatedPositions[i * 3] =
          x * Math.cos(rotation) - y * Math.sin(rotation);
        rotatedPositions[i * 3 + 1] =
          x * Math.sin(rotation) + y * Math.cos(rotation);
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array(rotatedPositions),
        gl.STATIC_DRAW
      );

      gl.drawArrays(gl.TRIANGLES, 0, numStarVertices);
      requestAnimationFrame(animate);
    };
    animate();

    return () => {
      gl.deleteProgram(program);
      gl.deleteBuffer(positionBuffer);
      gl.deleteBuffer(colorBuffer);
      gl.deleteVertexArray(vao);
    };
  }, []);

  useEffect(() => {
    if (userInput.toUpperCase() === targetText) {
      setIsCorrect(true);
      recordSuccess();
    }
  }, [userInput, recordSuccess, targetText]);

  return (
    <div className="w-full h-screen flex flex-col items-center bg-slate-100">
      {!isCorrect ? (
        <>
          <h1 className="text-2xl font-bold mt-4 mb-2">WebGL Hard Challenge</h1>
          <p className="text-lg text-gray-700 mb-4">
            Identify the celestial body spinning before you. What is its name?
          </p>

          <div className="flex-1 w-full relative overflow-hidden flex justify-center items-center">
            <div
              ref={containerRef}
              className="relative"
              style={{
                width: 800,
                height: 600,
                transform: `scale(${scale})`,
                transformOrigin: "center",
              }}
            >
              <canvas
                ref={canvasRef}
                width={800}
                height={600}
                className="bg-black rounded-lg shadow-lg"
              />
            </div>
          </div>

          <div className="w-full max-w-md px-4 mb-8">
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="What shape do you see?"
              className="w-full p-3 text-lg rounded-lg border-2 border-gray-300 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center justify-center min-h-screen w-full">
          <div className="text-4xl mb-4 text-green-600">
            🎉 Congratulations! 🎉
          </div>
          <div className="text-2xl mb-8">
            You've successfully identified the star!
          </div>
          <div className="text-xl bg-yellow-100 p-6 rounded-lg border-2 border-yellow-400">
            The secret password is:{" "}
            <span className="font-bold">{PASSWORD_WebGLTextHard}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default WebGLTextHard;
