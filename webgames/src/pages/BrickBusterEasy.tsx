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
const BRICK_ROWS = 2; // Easy: Reduced rows
const BRICK_COLS = 4; // Easy: Reduced columns
const BRICK_PADDING = 10;
const BRICK_WIDTH =
  (CANVAS_WIDTH - (BRICK_COLS + 1) * BRICK_PADDING) / BRICK_COLS;
const BRICK_HEIGHT = 30;
const PADDLE_WIDTH = 120; // Easy: Larger paddle
const PADDLE_HEIGHT = 15;
const BALL_RADIUS = 8;
const BALL_SPEED = 2.5; // Easy: Slower ball
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
      dx: BALL_SPEED,
      dy: -BALL_SPEED,
      radius: BALL_RADIUS,
    },
    bricks,
    score: 0,
  };
};

export const PASSWORD_BrickBusterEasy = "BrickBusterBeginner2025";
export const TASK_ID_BrickBusterEasy = "brick-buster-easy";

const BrickBusterEasy = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameStateRef = useRef<GameState>(createInitialState());
  const [gameStarted, setGameStarted] = useState(false);
  const [displayScore, setDisplayScore] = useState(0);
  const [password, setPassword] = useState("");
  const { recordSuccess } = useTaskAnalytics(TASK_ID_BrickBusterEasy);
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

    // Calculate paddle velocity
    const paddleVelocity =
      (state.paddle.x - state.paddle.lastX) / (deltaTime || 16.67);
    state.paddle.lastX = state.paddle.x;

    // Clear canvas
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Update ball position
    const ball = state.ball;
    const newX = ball.x + ball.dx;
    const newY = ball.y + ball.dy;

    // Wall collisions
    if (newX + ball.radius > CANVAS_WIDTH || newX - ball.radius < 0) {
      ball.dx = -ball.dx;
    }
    if (newY - ball.radius < 0) {
      ball.dy = -ball.dy;
    }

    // Paddle collision with improved physics
    if (
      newY + ball.radius > state.paddle.y &&
      newY + ball.radius < state.paddle.y + state.paddle.height &&
      newX > state.paddle.x &&
      newX < state.paddle.x + state.paddle.width
    ) {
      // Calculate hit position relative to paddle center (-1 to 1)
      const hitPos =
        (newX - (state.paddle.x + state.paddle.width / 2)) /
        (state.paddle.width / 2);

      // Calculate bounce angle (maximum ±60 degrees)
      const angle = hitPos * MAX_BALL_ANGLE;

      // Calculate new velocity components
      const speed = Math.sqrt(ball.dx * ball.dx + ball.dy * ball.dy);
      ball.dx = speed * Math.sin(angle);
      ball.dy = -speed * Math.cos(angle);

      // Add some of the paddle's velocity to the ball
      ball.dx += paddleVelocity * 0.3; // 30% of paddle velocity

      // Ensure minimum vertical velocity to prevent horizontal bouncing
      const minVerticalSpeed = speed * 0.5;
      if (Math.abs(ball.dy) < minVerticalSpeed) {
        ball.dy = ball.dy < 0 ? -minVerticalSpeed : minVerticalSpeed;
      }

      // Cap maximum speed
      const currentSpeed = Math.sqrt(ball.dx * ball.dx + ball.dy * ball.dy);
      if (currentSpeed > BALL_SPEED * 1.5) {
        const scale = (BALL_SPEED * 1.5) / currentSpeed;
        ball.dx *= scale;
        ball.dy *= scale;
      }
    }

    // Game over condition
    if (newY + ball.radius > CANVAS_HEIGHT) {
      setGameStarted(false);
      return;
    }

    ball.x = newX;
    ball.y = newY;

    // Brick collisions
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
          setPassword(PASSWORD_BrickBusterEasy);
          recordSuccess();
          setGameStarted(false);
          return;
        }
      }
    });

    // Draw game objects
    // Draw paddle
    ctx.fillStyle = "#0066cc";
    ctx.fillRect(state.paddle.x, state.paddle.y, PADDLE_WIDTH, PADDLE_HEIGHT);

    // Draw ball
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = "#ff3300";
    ctx.fill();
    ctx.closePath();

    // Draw bricks
    state.bricks.forEach((brick) => {
      if (!brick.visible) return;
      ctx.fillStyle = "#33cc33";
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
          <h1 className="text-3xl font-bold mb-2">BrickBuster (Easy)</h1>
          <p className="text-gray-600 mb-4">
            Break all the bricks to reveal the password! This is an easier
            version with fewer bricks, a larger paddle and a slower ball.
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
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
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

export default BrickBusterEasy;
