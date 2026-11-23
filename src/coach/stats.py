import psycopg

from coach import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id serial PRIMARY KEY,
    user_id text NOT NULL,
    problem_id text NOT NULL,
    trap text NOT NULL,
    is_correct boolean NOT NULL,
    confidence smallint NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    created_at timestamptz NOT NULL DEFAULT now()
)
"""

SUMMARY = """
SELECT trap,
       count(*),
       avg(is_correct::int),
       avg(confidence) / 100,
       avg(power(confidence / 100.0 - is_correct::int, 2))
FROM attempts
WHERE user_id = %s
GROUP BY ROLLUP (trap)
ORDER BY grouping(trap) DESC, avg(is_correct::int), trap
"""


CALIBRATION = """
SELECT least(confidence / 20, 4) AS bin, count(*), avg(confidence) / 100, avg(is_correct::int)
FROM attempts
WHERE user_id = %s
GROUP BY bin
ORDER BY bin
"""


def connect():
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    conn.execute(SCHEMA)
    return conn


def save(user_id, problem_id, trap, is_correct, confidence):
    with connect() as conn:
        conn.execute(
            "INSERT INTO attempts (user_id, problem_id, trap, is_correct, confidence) VALUES (%s, %s, %s, %s, %s)",
            (user_id, problem_id, trap, is_correct, confidence),
        )


def solved(user_id):
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT problem_id FROM attempts WHERE user_id = %s", (user_id,))
        return [problem_id for (problem_id,) in rows]


def summary(user_id):
    with connect() as conn:
        total, *by_trap = conn.execute(SUMMARY, (user_id,)).fetchall()
        bins = conn.execute(CALIBRATION, (user_id,)).fetchall()
    if total[1] == 0:
        return {"attempts": 0}

    return {
        "attempts": total[1],
        "accuracy": round(float(total[2]), 2),
        "mean_confidence": round(float(total[3]), 2),
        "brier": round(float(total[4]), 3),
        "by_trap": [
            {"trap": trap, "attempts": n, "accuracy": round(float(acc), 2), "mean_confidence": round(float(conf), 2)}
            for trap, n, acc, conf, _ in by_trap
        ],
        "calibration": [
            {"confidence_range": f"{b * 20}–{b * 20 + 20}%", "attempts": n,
             "mean_confidence": round(float(conf), 2), "accuracy": round(float(acc), 2)}
            for b, n, conf, acc in bins
        ],
    }
