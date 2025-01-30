import { D1Database } from "@cloudflare/workers-types";

interface Env {
  DB: D1Database;
}

export interface TaskCompletion {
  taskId: string;
  completionTime: string;
  userId: string;
  startTime: string;
}

export const onRequestPost = async (context: {
  request: Request;
  env: Env;
}) => {
  try {
    const data: TaskCompletion = await context.request.json();
    const { taskId, completionTime, userId, startTime } = data;

    await context.env.DB.prepare(
      `INSERT INTO completions (task_id, start_time, completion_time, user_agent, ip_address, user_id)
       VALUES (?, ?, ?, ?, ?, ?)`
    )
      .bind(
        taskId,
        startTime,
        completionTime,
        context.request.headers.get("User-Agent") || "",
        context.request.headers.get("CF-Connecting-IP") || "",
        userId || ""
      )
      .run();

    return new Response(JSON.stringify({ success: true }), {
      headers: { "Content-Type": "application/json" },
      status: 200,
    });
  } catch (error) {
    console.error("Error saving task completion:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: "Failed to save task completion",
      }),
      {
        headers: { "Content-Type": "application/json" },
        status: 500,
      }
    );
  }
};
