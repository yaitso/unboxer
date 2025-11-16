import asyncpg
from os import environ
from typing import Optional


class RolloutsDB:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or environ.get("POSTGRES")
        if not self.dsn:
            raise ValueError("POSTGRES connection string not found in environment")
        self.pool: Optional[asyncpg.Pool] = None

    @staticmethod
    async def migrate(dsn: Optional[str] = None):
        """run migration once before training starts"""
        connection_string = dsn or environ.get("POSTGRES")
        if not connection_string:
            raise ValueError("POSTGRES connection string not found in environment")

        conn = await asyncpg.connect(connection_string)
        try:
            await conn.execute("DROP SCHEMA IF EXISTS unboxer CASCADE")
            await conn.execute("CREATE SCHEMA unboxer")
            await conn.execute("""
                CREATE TABLE unboxer.rollouts (
                    id SERIAL PRIMARY KEY,
                    train_run INTEGER DEFAULT NULL,
                    train_step INTEGER NOT NULL,
                    train_commit TEXT DEFAULT NULL,
                    rollout_name TEXT NOT NULL,
                    blackbox TEXT NOT NULL,
                    reward REAL DEFAULT 0.0,
                    solved BOOLEAN DEFAULT FALSE,
                    num_turns INTEGER DEFAULT 0,
                    trajectory JSONB DEFAULT NULL,
                    logs JSONB DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE INDEX idx_rollouts_train_run ON unboxer.rollouts(train_run);
                CREATE INDEX idx_rollouts_solved ON unboxer.rollouts(solved);
                CREATE INDEX idx_rollouts_created_at ON unboxer.rollouts(created_at DESC);
            """)
        finally:
            await conn.close()

    async def connect(self):
        self.pool = await asyncpg.create_pool(self.dsn)
        await self.init_schema()
        return self

    async def init_schema(self):
        """ensure schema exists (idempotent, no migration)"""
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE SCHEMA IF NOT EXISTS unboxer")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS unboxer.rollouts (
                    id SERIAL PRIMARY KEY,
                    train_run INTEGER DEFAULT NULL,
                    train_step INTEGER NOT NULL,
                    train_commit TEXT DEFAULT NULL,
                    rollout_name TEXT NOT NULL,
                    blackbox TEXT NOT NULL,
                    reward REAL DEFAULT 0.0,
                    solved BOOLEAN DEFAULT FALSE,
                    num_turns INTEGER DEFAULT 0,
                    trajectory JSONB DEFAULT NULL,
                    logs JSONB DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rollouts_train_run ON unboxer.rollouts(train_run);
                CREATE INDEX IF NOT EXISTS idx_rollouts_solved ON unboxer.rollouts(solved);
                CREATE INDEX IF NOT EXISTS idx_rollouts_created_at ON unboxer.rollouts(created_at DESC);
            """)

    async def add_rollout(
        self,
        train_step: int,
        rollout_name: str,
        blackbox: str,
        reward: float = 0.0,
        train_run: Optional[int] = None,
        train_commit: Optional[str] = None,
    ) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO unboxer.rollouts (train_run, train_step, train_commit, rollout_name, blackbox, reward)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                train_run,
                train_step,
                train_commit,
                rollout_name,
                blackbox,
                reward,
            )
            return row["id"]

    async def update_reward(self, rollout_id: int, reward: float):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE unboxer.rollouts SET reward = $1 WHERE id = $2",
                reward,
                rollout_id,
            )

    async def update_trajectory(
        self,
        rollout_id: int,
        trajectory: list,
        num_turns: int,
        solved: bool,
    ):
        import json

        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE unboxer.rollouts 
                   SET trajectory = $1, num_turns = $2, solved = $3 
                   WHERE id = $4""",
                json.dumps(trajectory),
                num_turns,
                solved,
                rollout_id,
            )

    async def append_log(
        self,
        rollout_id: int,
        event: str,
        data: dict,
    ):
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "data": data,
        }
        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE unboxer.rollouts 
                   SET logs = logs || $1::jsonb 
                   WHERE id = $2""",
                json.dumps([log_entry]),
                rollout_id,
            )

    async def get_rollout_window(self, window_size: int = 100) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH blackbox_stats AS (
                    SELECT
                        train_run,
                        blackbox,
                        AVG(reward) as mean_reward,
                        MAX(created_at) as latest_created_at
                    FROM unboxer.rollouts
                    GROUP BY train_run, blackbox
                ),
                latest_per_blackbox AS (
                    SELECT
                        blackbox,
                        mean_reward,
                        latest_created_at,
                        ROW_NUMBER() OVER (PARTITION BY blackbox ORDER BY train_run DESC NULLS LAST) as rn
                    FROM blackbox_stats
                )
                SELECT blackbox, mean_reward
                FROM latest_per_blackbox
                WHERE rn = 1
                ORDER BY latest_created_at DESC
                LIMIT $1
                """,
                window_size,
            )
            return [
                {"blackbox": row["blackbox"], "mean_reward": float(row["mean_reward"])}
                for row in rows
            ]

    async def get_next_train_run(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(train_run) as max_run FROM unboxer.rollouts"
            )
            max_run = row["max_run"]
            return (max_run + 1) if max_run is not None else 1

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
