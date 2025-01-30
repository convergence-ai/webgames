CREATE TABLE IF NOT EXISTS completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    completion_time DATETIME NOT NULL,
    user_agent TEXT,
    ip_address TEXT,
    user_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_task_id ON completions(task_id);
CREATE INDEX idx_created_at ON completions(created_at);

INSERT INTO completions (task_id, start_time, completion_time, user_agent, ip_address, user_id)
VALUES ('home', '2025-01-30 12:00:00', '2025-01-30 12:00:01', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', '127.0.0.1', '1234567890');

