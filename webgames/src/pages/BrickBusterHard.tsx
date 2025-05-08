import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface GameState {
  paddle: {
    x: number;
    y: number;
    width: number;
    height: number;
    lastX: number;
  };
  ball: {
    x: number;
    y: number;
    dx: number;
    dy: number;
    radius: number;
  };
  bricks: {
    x: number;
    y: number;
    width: number;
    height: number;
    visible: boolean;
  }[];
  score: number;
}

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const BRICK_ROWS = 7; // Hard: Increased rows
const BRICK_COLS = 10; // Hard: Increased columns
const BRICK_PADDING = 10;
const BRICK_WIDTH =
  (CANVAS_WIDTH - (BRICK_COLS + 1) * BRICK_PADDING) / BRICK_COLS;
const BRICK_HEIGHT = 20; // Hard: Smaller bricks
const PADDLE_WIDTH = 80; // Hard: Smaller paddle
const PADDLE_HEIGHT = 15;
const BALL_RADIUS = 7; // Hard: Smaller ball
const BALL_SPEED = 4; // Hard: Faster ball
const MAX_BALL_ANGLE = Math.PI / 3;

const createInitialState = (): GameState => {
  const bricks = [];
  for (let row = 0; row < BRICK_ROWS; row++) {
    for (let col = 0; col < BRICK_COLS; col++) {
      bricks.push({
        x: col * (BRICK_WIDTH + BRICK_PADDING) + BRICK_PADDING,
        y: row * (BRICK_HEIGHT + BRICK_PADDING) + BRICK_PADDING + 50,
        width: BRICK_WIDTH,
        height: BRICK_HEIGHT,
        visible: true,
      });
    }
  }

  const paddleX = CANVAS_WIDTH / 2 - PADDLE_WIDTH / 2;

  return {
    paddle: {
      x: paddleX,
      y: CANVAS_HEIGHT - 30,
      width: PADDLE_WIDTH,
      height: PADDLE_HEIGHT,
      lastX: paddleX,
    },
    ball: {
      x: CANVAS_WIDTH / 2,
      y: CANVAS_HEIGHT - 50,
      dx: BALL_SPEED * (Math.random() > 0.5 ? 1 : -1), // Random initial direction
      dy: -BALL_SPEED,
      radius: BALL_RADIUS,
    },
    bricks,
    score: 0,
  };
};

export const PASSWORD_BrickBusterHard = "BrickBusterExpert2025";
export const TASK_ID_BrickBusterHard = "brick-buster-hard";

const BrickBusterHard = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameStateRef = useRef<GameState>(createInitialState());
  const [gameStarted, setGameStarted] = useState(false);
  const [displayScore, setDisplayScore] = useState(0);
  const [password, setPassword] = useState("");
  const { recordSuccess } = useTaskAnalytics(TASK_ID_BrickBusterHard);
  const requestRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);

  const updateGame = (timestamp: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const deltaTime = timestamp - lastTimeRef.current;
    lastTimeRef.current = timestamp;

    const state = gameStateRef.current;

    const paddleVelocity =
      (state.paddle.x - state.paddle.lastX) / (deltaTime || 16.67);
    state.paddle.lastX = state.paddle.x;

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    const ball = state.ball;
    const newX = ball.x + ball.dx;
    const newY = ball.y + ball.dy;

    if (newX + ball.radius > CANVAS_WIDTH || newX - ball.radius < 0) {
      ball.dx = -ball.dx;
    }
    if (newY - ball.radius < 0) {
      ball.dy = -ball.dy;
    }

    if (
      newY + ball.radius > state.paddle.y &&
      newY + ball.radius < state.paddle.y + state.paddle.height &&
      newX > state.paddle.x &&
      newX < state.paddle.x + state.paddle.width
    ) {
      const hitPos =
        (newX - (state.paddle.x + state.paddle.width / 2)) /
        (state.paddle.width / 2);
      const angle = hitPos * MAX_BALL_ANGLE;
      const speed = Math.sqrt(ball.dx * ball.dx + ball.dy * ball.dy);
      ball.dx = speed * Math.sin(angle);
      ball.dy = -speed * Math.cos(angle);
      ball.dx += paddleVelocity * 0.4; // Hard: Paddle has more influence

      const minVerticalSpeed = speed * 0.4;
      if (Math.abs(ball.dy) < minVerticalSpeed) {
        ball.dy = ball.dy < 0 ? -minVerticalSpeed : minVerticalSpeed;
      }

      const currentSpeed = Math.sqrt(ball.dx * ball.dx + ball.dy * ball.dy);
      const maxSpeed = BALL_SPEED * 1.8; // Hard: Higher max speed
      if (currentSpeed > maxSpeed) {
        const scale = maxSpeed / currentSpeed;
        ball.dx *= scale;
        ball.dy *= scale;
      }
    }

    if (newY + ball.radius > CANVAS_HEIGHT) {
      setGameStarted(false);
      return;
    }

    ball.x = newX;
    ball.y = newY;

    state.bricks.forEach((brick) => {
      if (!brick.visible) return;
      if (
        ball.x > brick.x &&
        ball.x < brick.x + brick.width &&
        ball.y > brick.y &&
        ball.y < brick.y + brick.height
      ) {
        brick.visible = false;
        ball.dy = -ball.dy;
        state.score += 1;
        setDisplayScore(state.score);
        if (state.score === BRICK_ROWS * BRICK_COLS) {
          setPassword(PASSWORD_BrickBusterHard);
          recordSuccess();
          setGameStarted(false);
          return;
        }
      }
    });

    ctx.fillStyle = "#004488"; // Hard: Darker paddle
    ctx.fillRect(state.paddle.x, state.paddle.y, PADDLE_WIDTH, PADDLE_HEIGHT);

    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = "#cc0000"; // Hard: Darker ball
    ctx.fill();
    ctx.closePath();

    state.bricks.forEach((brick) => {
      if (!brick.visible) return;
      ctx.fillStyle = "#228822"; // Hard: Darker bricks
      ctx.fillRect(brick.x, brick.y, brick.width, brick.height);
    });

    requestRef.current = requestAnimationFrame(updateGame);
  };

  useEffect(() => {
    if (!gameStarted) {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
      return;
    }
    lastTimeRef.current = performance.now();
    requestRef.current = requestAnimationFrame(updateGame);
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [gameStarted]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      gameStateRef.current.paddle.x = Math.max(
        0,
        Math.min(x - PADDLE_WIDTH / 2, CANVAS_WIDTH - PADDLE_WIDTH)
      );
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  const handleStartGame = () => {
    gameStateRef.current = createInitialState();
    setDisplayScore(0);
    setPassword("");
    setGameStarted(true);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-4xl">
        <div className="text-center mb-4">
          <h1 className="text-3xl font-bold mb-2">BrickBuster (Hard)</h1>
          <p className="text-gray-600 mb-4">
            Break all the bricks to reveal the password! This is a harder
            version with more bricks, a smaller paddle, smaller ball, and faster
            ball speed.
          </p>
          <div className="mb-4">
            <p className="text-xl">Score: {displayScore}</p>
            {password && (
              <p className="text-green-600 font-bold mt-2">
                Password: {password}
              </p>
            )}
          </div>
          {!gameStarted && (
            <button
              onClick={handleStartGame}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded"
            >
              {displayScore > 0 ? "Restart Game" : "Start Game"}
            </button>
          )}
        </div>
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          className="border border-gray-300 rounded-lg mx-auto"
        />
      </div>
    </div>
  );
};

export default BrickBusterHard;
