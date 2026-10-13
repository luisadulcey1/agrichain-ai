
"""
AgriChain AI
Resolution Management Database

SQLite database for storing complaint resolution records.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "data" / "agrichain_resolutions.db"

VALID_STATUSES = (
    "Pending Review",
    "In Progress",
    "Pending Approval",
    "Approved",
    "Resolved",
    "Escalated",
    "Closed",
)


def get_connection():
    """
    Create a SQLite connection.
    """

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    return connection


def current_timestamp():
    """
    Return the current UTC timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")


# ---------------------------------------------------------
# INITIALIZE DATABASE
# ---------------------------------------------------------

def initialize_database():
    """
    Create the complaints table if it does not exist.
    """

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_text TEXT NOT NULL,
                predicted_category TEXT,
                validated_category TEXT,
                confidence REAL,
                status TEXT NOT NULL DEFAULT 'Pending Review',
                resolution_plan TEXT,
                customer_email TEXT,
                email_approved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_complaints_status
            ON complaints(status)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_complaints_created_at
            ON complaints(created_at)
            """
        )

        connection.commit()


# ---------------------------------------------------------
# CREATE COMPLAINT
# ---------------------------------------------------------

def create_complaint(
    complaint_text,
    predicted_category=None,
    validated_category=None,
    confidence=None,
    status="Pending Review",
    resolution_plan=None,
    customer_email=None,
    email_approved=False
):
    """
    Save a new complaint and return its database ID.
    """

    initialize_database()

    if not complaint_text or not complaint_text.strip():
        raise ValueError(
            "Complaint text cannot be empty."
        )

    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid complaint status: {status}"
        )

    if confidence is not None:
        confidence = float(confidence)

        if not 0 <= confidence <= 1:
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

    if resolution_plan is not None and not isinstance(
        resolution_plan, str
    ):
        resolution_plan = json.dumps(
            resolution_plan,
            ensure_ascii=False
        )

    timestamp = current_timestamp()

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO complaints (
                complaint_text,
                predicted_category,
                validated_category,
                confidence,
                status,
                resolution_plan,
                customer_email,
                email_approved,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                complaint_text.strip(),
                predicted_category,
                validated_category,
                confidence,
                status,
                resolution_plan,
                customer_email,
                int(bool(email_approved)),
                timestamp,
                timestamp
            )
        )

        connection.commit()

        return cursor.lastrowid


# ---------------------------------------------------------
# READ ALL COMPLAINTS
# ---------------------------------------------------------

def get_all_complaints():
    """
    Return all complaints, newest first.
    """

    initialize_database()

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM complaints
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ---------------------------------------------------------
# READ ONE COMPLAINT
# ---------------------------------------------------------

def get_complaint_by_id(complaint_id):
    """
    Return a complaint by its database ID.
    """

    initialize_database()

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT *
            FROM complaints
            WHERE id = ?
            """,
            (complaint_id,)
        ).fetchone()

    if row is None:
        return None

    return dict(row)


# ---------------------------------------------------------
# UPDATE COMPLAINT STATUS
# ---------------------------------------------------------

def update_complaint_status(
    complaint_id,
    new_status
):
    """
    Update the resolution status of a complaint.
    """

    initialize_database()

    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid complaint status: {new_status}"
        )

    with get_connection() as connection:

        cursor = connection.execute(
            """
            UPDATE complaints
            SET status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                new_status,
                current_timestamp(),
                complaint_id
            )
        )

        connection.commit()

        return cursor.rowcount > 0


# ---------------------------------------------------------
# UPDATE RESOLUTION PLAN
# ---------------------------------------------------------

def update_resolution_plan(
    complaint_id,
    resolution_plan
):
    """
    Update the proposed resolution plan.
    """

    initialize_database()

    if not isinstance(resolution_plan, str):
        resolution_plan = json.dumps(
            resolution_plan,
            ensure_ascii=False
        )

    with get_connection() as connection:

        cursor = connection.execute(
            """
            UPDATE complaints
            SET resolution_plan = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                resolution_plan,
                current_timestamp(),
                complaint_id
            )
        )

        connection.commit()

        return cursor.rowcount > 0


# ---------------------------------------------------------
# SAVE CUSTOMER EMAIL
# ---------------------------------------------------------

def save_customer_email(
    complaint_id,
    customer_email
):
    """
    Save or edit a customer email draft.

    Editing an email automatically removes
    any previous email approval.
    """

    initialize_database()

    if not customer_email or not customer_email.strip():
        raise ValueError(
            "Customer email cannot be empty."
        )

    with get_connection() as connection:

        cursor = connection.execute(
            """
            UPDATE complaints
            SET customer_email = ?,
                email_approved = 0,
                status = 'Pending Approval',
                updated_at = ?
            WHERE id = ?
            """,
            (
                customer_email.strip(),
                current_timestamp(),
                complaint_id
            )
        )

        connection.commit()

        return cursor.rowcount > 0


# ---------------------------------------------------------
# APPROVE CUSTOMER EMAIL
# ---------------------------------------------------------

def approve_customer_email(complaint_id):
    """
    Mark an existing email draft as approved.

    This does not send the email.
    """

    initialize_database()

    with get_connection() as connection:

        cursor = connection.execute(
            """
            UPDATE complaints
            SET email_approved = 1,
                status = 'Approved',
                updated_at = ?
            WHERE id = ?
              AND customer_email IS NOT NULL
              AND LENGTH(TRIM(customer_email)) > 0
            """,
            (
                current_timestamp(),
                complaint_id
            )
        )

        connection.commit()

        return cursor.rowcount > 0


# ---------------------------------------------------------
# DATABASE STATISTICS
# ---------------------------------------------------------

def get_database_statistics():
    """
    Return complaint management metrics.
    """

    initialize_database()

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_complaints,

                SUM(
                    CASE
                        WHEN status = 'Resolved'
                        THEN 1
                        ELSE 0
                    END
                ) AS resolved_complaints,

                SUM(
                    CASE
                        WHEN status = 'Escalated'
                        THEN 1
                        ELSE 0
                    END
                ) AS escalated_complaints,

                SUM(
                    CASE
                        WHEN email_approved = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS approved_emails

            FROM complaints
            """
        ).fetchone()

    return {
        "total_complaints": row["total_complaints"] or 0,
        "resolved_complaints": row["resolved_complaints"] or 0,
        "escalated_complaints": row["escalated_complaints"] or 0,
        "approved_emails": row["approved_emails"] or 0
    }


# ---------------------------------------------------------
# INITIALIZE ON DIRECT EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":

    initialize_database()

    print("AgriChain AI database initialized successfully.")

    print(f"Database location: {DATABASE_PATH}")
