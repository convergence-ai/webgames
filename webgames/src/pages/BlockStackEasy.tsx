import {
  Bodies,
  Body,
  Engine,
  Events,
  Mouse,
  MouseConstraint,
  Query,
  Render,
  Runner,
  World,
} from "matter-js";
import React, { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_BlockStackEasy = "EquilibriumLite"; // Easy variant password
export const TASK_ID_BlockStackEasy = "block-stack-easy";
const BlockStackEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_BlockStackEasy);
  const scene = useRef<HTMLDivElement>(null);
  const engine = useRef(Engine.create());
  const [isComplete, setIsComplete] = useState(false);
  const [highestY, setHighestY] = useState<number>(0);
  const [targetLineY, setTargetLineY] = useState<number>(0);
  const [isAboveLine, setIsAboveLine] = useState(false);
  const [timeAbove, setTimeAbove] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const startTimeRef = useRef<number | null>(null);

  // Timer duration for easy variant
  const REQUIRED_TIME_ABOVE = 1; // seconds

  useEffect(() => {
    let intervalId: number | null = null;

    if (isAboveLine && !isComplete) {
      if (!startTimeRef.current) {
        startTimeRef.current = Date.now();
      }

      intervalId = window.setInterval(() => {
        const currentTime = (Date.now() - startTimeRef.current!) / 1000;
        setTimeAbove(currentTime);

        if (currentTime >= REQUIRED_TIME_ABOVE) {
          setIsComplete(true);
          recordSuccess();
          if (intervalId) {
            clearInterval(intervalId);
          }
        }
      }, 50);
    } else {
      startTimeRef.current = null;
      setTimeAbove(0);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isAboveLine, isComplete, recordSuccess]);

  useEffect(() => {
    if (!scene.current) return;

    const cw = window.innerWidth;
    const ch = window.innerHeight;
    const targetY = ch * 0.5; // Higher target line for easiness
    setTargetLineY(targetY);

    const render = Render.create({
      element: scene.current,
      engine: engine.current,
      options: {
        width: cw,
        height: ch,
        wireframes: false,
        background: "#f0f0f0",
      },
    });

    const walls = [
      Bodies.rectangle(cw / 2, ch + 30, cw, 60, {
        isStatic: true,
        render: { fillStyle: "#666" },
      }),
      Bodies.rectangle(-30, ch / 2, 60, ch, {
        isStatic: true,
        render: { fillStyle: "#666" },
      }),
      Bodies.rectangle(cw + 30, ch / 2, 60, ch, {
        isStatic: true,
        render: { fillStyle: "#666" },
      }),
    ];

    const targetLine = Bodies.rectangle(cw / 2, targetY, cw, 2, {
      isStatic: true,
      isSensor: true,
      render: { fillStyle: "#00ff00" }, // Green line for easy
    });

    const boxes: Body[] = [];
    const boxSize = Math.min(Math.min(cw, ch) * 0.12, 90); // Slightly larger boxes
    const boxesPerRow = Math.min(Math.floor(cw / (boxSize * 1.5)), 4); // Fewer boxes per row
    const numRows = Math.min(Math.floor((ch - targetY) / (boxSize * 1.5)), 2); // Fewer rows
    const numBoxes = boxesPerRow * numRows;
    const startX = cw / 2 - (boxSize * boxesPerRow) / 2;
    const startY = ch - boxSize * 1.5; // Start boxes a bit higher

    for (let i = 0; i < numBoxes; i++) {
      boxes.push(
        Bodies.rectangle(
          startX + (i % boxesPerRow) * (boxSize * 1.2),
          startY - Math.floor(i / boxesPerRow) * (boxSize * 1.2),
          boxSize,
          boxSize,
          {
            render: {
              fillStyle: `hsl(${Math.random() * 360}, 70%, 60%)`, // Brighter colors
              strokeStyle: "#000",
              lineWidth: 2,
            },
            chamfer: { radius: boxSize * 0.08 },
            density: 0.15, // Lighter boxes
            friction: 0.7,
            restitution: 0.1,
          }
        )
      );
    }

    World.add(engine.current.world, [...walls, ...boxes, targetLine]);

    const mouse = Mouse.create(render.canvas);
    const mouseConstraint = MouseConstraint.create(engine.current, {
      mouse: mouse,
      constraint: {
        stiffness: 0.2,
        render: {
          visible: false,
        },
      },
    }) as MouseConstraint & { body?: Body };

    World.add(engine.current.world, mouseConstraint);
    render.mouse = mouse;

    Events.on(mouseConstraint, "mousedown", () => {
      const bodies = Query.point(engine.current.world.bodies, mouse.position);
      if (bodies.length > 0 && !bodies[0].isStatic) {
        setIsDragging(true);
      }
    });

    Events.on(mouseConstraint, "mouseup", () => {
      setIsDragging(false);
    });

    Events.on(mouseConstraint, "mousemove", () => {
      if (mouseConstraint.body) {
        setIsDragging(true);
      }
    });

    Events.on(engine.current, "afterUpdate", () => {
      const currentBoxes = engine.current.world.bodies.filter(
        (body) => !body.isStatic && body !== mouseConstraint.body
      );

      if (!isDragging) {
        const boxAboveLine = currentBoxes.some((box) => {
          const vertices = box.vertices;
          return vertices.every((vertex) => vertex.y < targetY);
        });
        setIsAboveLine(boxAboveLine);
      } else {
        setIsAboveLine(false);
      }

      const currentHighestY =
        currentBoxes.length > 0
          ? Math.min(
              ...currentBoxes.flatMap((box) => box.vertices.map((v) => v.y))
            )
          : ch; // Default to bottom if no boxes

      setHighestY(Math.round(currentHighestY));
    });

    const runner = Runner.create();
    Runner.run(runner, engine.current);
    Render.run(render);

    return () => {
      Render.stop(render);
      Runner.stop(runner);
      World.clear(engine.current.world, false);
      Engine.clear(engine.current);
      render.canvas.remove();
    };
  }, []);

  return (
    <div style={{ position: "relative" }}>
      <div ref={scene} style={{ width: "100%", height: "100vh" }} />
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(255, 255, 255, 0.9)",
          padding: "15px",
          borderRadius: "8px",
          fontSize: "16px",
        }}
      >
        {!isComplete ? (
          <div>
            <div>
              Stack blocks above the green line and keep them there for{" "}
              {REQUIRED_TIME_ABOVE} second! (Easy)
            </div>
            <div style={{ marginTop: "10px", fontFamily: "monospace" }}>
              Target Line Y: {Math.round(targetLineY)}
              <br />
              Highest Box Y: {highestY}
              <br />
              Mouse Status: {isDragging ? "Dragging" : "Not Dragging"}
              <br />
              {isAboveLine && `Time above: ${timeAbove.toFixed(1)}s`}
            </div>
          </div>
        ) : (
          <span style={{ color: "#4CAF50", fontWeight: "bold" }}>
            Challenge Complete! 🎉 Password: {PASSWORD_BlockStackEasy}
          </span>
        )}
      </div>
    </div>
  );
};

export default BlockStackEasy;
