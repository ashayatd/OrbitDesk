"""Seed fictional employees, leave balances and a couple of tickets. Idempotent."""

from app.db.database import db_session, init_db

EMPLOYEES = [
    # id, name, email, role, department, manager_id
    ("E1001", "Priya Raman", "priya.raman@orbit.example", "employee", "Engineering", "E1005"),
    ("E1002", "Daniel Okafor", "daniel.okafor@orbit.example", "employee", "Sales", "E1006"),
    ("E1003", "Mei Lin", "mei.lin@orbit.example", "hr", "HR", None),
    ("E1004", "Tomas Novak", "tomas.novak@orbit.example", "it_admin", "IT", None),
    ("E1005", "Aisha Bello", "aisha.bello@orbit.example", "employee", "Engineering", None),
    ("E1006", "Lucas Martin", "lucas.martin@orbit.example", "employee", "Sales", None),
]

# employee_id -> {leave_type: (entitled, used)}
LEAVE = {
    "E1001": {"annual": (24, 9), "sick": (10, 2), "carry_over": (5, 1)},
    "E1002": {"annual": (24, 20.5), "sick": (10, 0), "carry_over": (3, 3)},
    "E1003": {"annual": (24, 4), "sick": (10, 1), "carry_over": (0, 0)},
    "E1004": {"annual": (24, 12), "sick": (10, 5), "carry_over": (5, 0)},
    "E1005": {"annual": (24, 15), "sick": (10, 3), "carry_over": (2, 2)},
    "E1006": {"annual": (24, 6), "sick": (10, 0), "carry_over": (5, 5)},
}

TICKETS = [
    (
        "TCK-1001",
        "E1001",
        "VPN-809 at hotel",
        "VPN fails with VPN-809 on hotel Wi-Fi",
        "VPN",
        "P3",
        "resolved",
    ),
    (
        "TCK-1002",
        "E1002",
        "Laptop blue screen",
        "Stop code 0x0000007E twice this week",
        "Hardware",
        "P3",
        "open",
    ),
]


def seed(path=None) -> None:
    init_db(path)
    with db_session(path) as conn:
        conn.executemany("INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?,?)", EMPLOYEES)
        for emp, balances in LEAVE.items():
            for leave_type, (entitled, used) in balances.items():
                conn.execute("INSERT OR IGNORE INTO leave_balances VALUES (?,?,?,?)", (emp, leave_type, entitled, used))
        conn.executemany(
            "INSERT OR IGNORE INTO tickets (id, employee_id, title, description, category, priority, status)"
            " VALUES (?,?,?,?,?,?,?)",
            TICKETS,
        )


if __name__ == "__main__":
    seed()
    print("seeded")
