"""Seeded synthetic HR data and sample policy documents. Run `python -m app.synthetic` to regenerate."""
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from .db import Database

SEED = 42
TODAY = date(2026, 11, 17)

DEPARTMENTS = [
    (1, "Engineering", "San Francisco", 9_000_000),
    (2, "Sales", "New York", 6_000_000),
    (3, "Marketing", "New York", 2_500_000),
    (4, "Customer Support", "Austin", 3_000_000),
    (5, "Finance", "Chicago", 2_000_000),
    (6, "HR", "Chicago", 1_500_000),
    (7, "Operations", "Austin", 3_500_000),
    (8, "Product", "San Francisco", 3_000_000),
]
# department_id: (titles with (title, level, salary_mid), attrition propensity)
TITLES = {
    1: [("Software Engineer", "IC", 135_000), ("Senior Engineer", "Senior", 170_000), ("Engineering Manager", "Manager", 195_000)],
    2: [("Account Executive", "IC", 95_000), ("Sales Manager", "Manager", 140_000), ("SDR", "Junior", 65_000)],
    3: [("Marketing Specialist", "IC", 80_000), ("Marketing Manager", "Manager", 120_000)],
    4: [("Support Agent", "Junior", 52_000), ("Support Lead", "Senior", 75_000)],
    5: [("Accountant", "IC", 85_000), ("Finance Manager", "Manager", 135_000)],
    6: [("HR Generalist", "IC", 78_000), ("HR Manager", "Manager", 115_000)],
    7: [("Operations Analyst", "IC", 78_000), ("Operations Manager", "Manager", 118_000)],
    8: [("Product Manager", "Senior", 150_000), ("Product Analyst", "IC", 105_000)],
}
ATTRITION = {1: 0.12, 2: 0.22, 3: 0.14, 4: 0.28, 5: 0.07, 6: 0.09, 7: 0.15, 8: 0.11}
REASONS = ["Resignation", "Resignation", "Resignation", "Layoff", "Retirement", "Dismissal"]
LOCATIONS = ["San Francisco", "New York", "Austin", "Chicago", "Remote"]
LEAVE_TYPES = ["Annual", "Sick", "Parental", "Unpaid", "Bereavement"]


def _rand_date(rng, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def generate_tables(n_employees: int = 600, seed: int = SEED) -> dict[str, tuple[list[str], list[tuple]]]:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    employees = []
    for eid in range(1, n_employees + 1):
        dept = rng.choices([d[0] for d in DEPARTMENTS], weights=[22, 16, 8, 18, 6, 4, 14, 12])[0]
        title, level, mid = rng.choice(TITLES[dept])
        hire = _rand_date(rng, date(2018, 1, 1), date(2026, 10, 1))
        salary = round(mid * rng.uniform(0.85, 1.2), -2)
        term_date = reason = None
        years = max((TODAY - hire).days / 365, 0.1)
        if rng.random() < min(0.85, 1 - (1 - ATTRITION[dept]) ** years):
            term_date = hire + timedelta(days=rng.randint(60, max(61, (TODAY - hire).days)))
            term_date = min(term_date, TODAY - timedelta(days=1))
            reason = rng.choice(REASONS)
        first, last = fake.first_name(), fake.last_name()
        employees.append((
            eid, first, last, f"{first}.{last}.{eid}@example.com".lower(), dept, title, level,
            rng.choice(["Female", "Male", "Non-binary"]) if rng.random() > 0.03 else "Undisclosed",
            hire, term_date, reason, "Terminated" if term_date else "Active",
            None, salary, rng.choices([1, 2, 3, 4, 5], weights=[4, 12, 40, 32, 12])[0], rng.choice(LOCATIONS),
        ))
    # assign managers: managers in the same department (any employee if none)
    mgrs = {d[0]: [e[0] for e in employees if e[4] == d[0] and e[6] == "Manager"] for d in DEPARTMENTS}
    fixed = []
    for e in employees:
        pool = [m for m in mgrs[e[4]] if m != e[0]]
        fixed.append(e[:12] + (rng.choice(pool) if pool and e[6] != "Manager" else None,) + e[13:])
    employees = fixed

    leave = []
    for eid, *_ in employees[:]:
        e = employees[eid - 1]
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            end_cap = e[9] or TODAY
            if end_cap <= e[8]:
                continue
            start = _rand_date(rng, e[8], end_cap)
            lt = rng.choices(LEAVE_TYPES, weights=[55, 28, 4, 8, 5])[0]
            days = {"Annual": rng.randint(1, 10), "Sick": rng.randint(1, 4), "Parental": rng.randint(20, 60),
                    "Unpaid": rng.randint(1, 15), "Bereavement": rng.randint(1, 5)}[lt]
            status = rng.choices(["Approved", "Pending", "Rejected"], weights=[85, 8, 7])[0]
            leave.append((len(leave) + 1, eid, lt, start, start + timedelta(days=days), days, status))

    payroll = []
    for e in employees:
        m = date(max(e[8].year, 2025), 1, 1) if e[8] < date(2025, 1, 1) else date(e[8].year, e[8].month, 1)
        last = e[9] or TODAY
        while m <= last and m <= TODAY:
            gross = round(e[13] / 12, 2)
            bonus = round(gross * rng.choice([0, 0, 0, 0.1, 0.25]), 2) if m.month in (3, 12) else 0.0
            ded = round((gross + bonus) * 0.22, 2)
            payroll.append((len(payroll) + 1, e[0], m, gross, bonus, ded, round(gross + bonus - ded, 2)))
            m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)

    reqs = []
    for rid in range(1, 121):
        dept = rng.choice([d[0] for d in DEPARTMENTS])
        title = rng.choice(TITLES[dept])[0]
        opened = _rand_date(rng, date(2024, 1, 1), date(2026, 10, 15))
        filled = opened + timedelta(days=rng.randint(15, 90))
        is_open = filled > TODAY or rng.random() < 0.15
        reqs.append((rid, dept, title, opened, None if is_open else filled, "Open" if is_open else "Filled",
                     rng.choice(["Referral", "LinkedIn", "Agency", "Careers Site", "Campus"])))

    return {
        "departments": (["department_id", "department_name", "location", "budget"], DEPARTMENTS),
        "employees": (
            ["employee_id", "first_name", "last_name", "email", "department_id", "job_title", "level", "gender",
             "hire_date", "termination_date", "termination_reason", "employment_status", "manager_id",
             "base_salary", "performance_rating", "location"], employees),
        "leave_requests": (["leave_id", "employee_id", "leave_type", "start_date", "end_date", "days", "status"], leave),
        "payroll": (["payroll_id", "employee_id", "pay_month", "gross_pay", "bonus", "deductions", "net_pay"], payroll),
        "job_requisitions": (["req_id", "department_id", "job_title", "opened_date", "filled_date", "status", "source"], reqs),
    }


DDL = {
    "departments": "department_id INTEGER, department_name VARCHAR, location VARCHAR, budget BIGINT",
    "employees": (
        "employee_id INTEGER, first_name VARCHAR, last_name VARCHAR, email VARCHAR, department_id INTEGER, "
        "job_title VARCHAR, level VARCHAR, gender VARCHAR, hire_date DATE, termination_date DATE, "
        "termination_reason VARCHAR, employment_status VARCHAR, manager_id INTEGER, base_salary DOUBLE, "
        "performance_rating INTEGER, location VARCHAR"
    ),
    "leave_requests": "leave_id INTEGER, employee_id INTEGER, leave_type VARCHAR, start_date DATE, end_date DATE, days INTEGER, status VARCHAR",
    "payroll": "payroll_id INTEGER, employee_id INTEGER, pay_month DATE, gross_pay DOUBLE, bonus DOUBLE, deductions DOUBLE, net_pay DOUBLE",
    "job_requisitions": "req_id INTEGER, department_id INTEGER, job_title VARCHAR, opened_date DATE, filled_date DATE, status VARCHAR, source VARCHAR",
}


def seed_database(db: Database, n_employees: int = 600) -> None:
    tables = generate_tables(n_employees)
    with db.lock:
        for name, (cols, rows) in tables.items():
            db.con.execute(f"DROP TABLE IF EXISTS {name}")
            db.con.execute(f"CREATE TABLE {name} ({DDL[name]})")
            if rows:
                ph = ",".join("?" * len(cols))
                db.con.executemany(f"INSERT INTO {name} VALUES ({ph})", rows)


POLICIES = {
    "leave_policy.md": """# Leave Policy

## Annual Leave
Full-time employees accrue 20 days of paid annual leave per calendar year, rising to 25 days after 5 years of service. Up to 5 unused days may be carried over into the next year; the rest expire on March 31.

## Sick Leave
Employees receive 10 paid sick days per year. A medical certificate is required for absences longer than 3 consecutive days.

## Parental Leave
Primary caregivers receive 16 weeks of fully paid parental leave; secondary caregivers receive 6 weeks. Leave must be taken within 12 months of the birth or adoption and requires 30 days' notice where possible.

## Bereavement Leave
Employees receive 5 paid days for the death of an immediate family member and 2 days for extended family.

## Unpaid Leave
Unpaid leave of up to 30 days may be approved by the department head and HR. Benefits continue for the first 30 days.
""",
    "remote_work_policy.md": """# Remote and Hybrid Work Policy

## Eligibility
Employees who have completed their 90-day probation may request hybrid work. Customer Support and Operations roles may be limited to on-site schedules by business need.

## Hybrid Schedule
Hybrid employees work from the office at least 3 days per week, including Tuesday and Thursday as team anchor days.

## Fully Remote
Fully remote arrangements require VP approval and are reviewed annually. Remote employees must be based in a country where the company has a legal entity.

## Equipment and Stipend
A one-time home-office stipend of 500 USD and a monthly internet stipend of 50 USD are provided to remote and hybrid employees.
""",
    "performance_review_policy.md": """# Performance Review Policy

## Cycle
Reviews run twice a year: mid-year check-in (June) and annual review (December). Ratings use a 1-5 scale where 3 means meets expectations.

## Calibration
Managers calibrate ratings within their department. No more than 15% of a team may receive a rating of 5.

## Performance Improvement Plans
Employees rated 1 or 2 may be placed on a 60-day Performance Improvement Plan (PIP) with written goals and bi-weekly check-ins. Failure to meet PIP goals may lead to termination.

## Promotions and Raises
Promotions require a rating of 4 or higher in the last two cycles. Merit increases range from 0% to 8% based on rating and budget.
""",
    "benefits_policy.md": """# Benefits Policy

## Health Insurance
Medical, dental and vision coverage begins on the first day of employment. The company covers 80% of employee premiums and 60% of dependent premiums.

## Retirement
The company matches 401(k) contributions up to 4% of salary, vesting immediately.

## Learning Budget
Each employee has an annual learning budget of 1,500 USD for courses, conferences and books, with manager approval.

## Wellness
A monthly wellness allowance of 40 USD is available for gym or mental-health apps.
""",
    "code_of_conduct.md": """# Code of Conduct

## Respectful Workplace
Harassment, discrimination and retaliation are prohibited. Reports can be made to HR, a manager, or the anonymous ethics hotline; all reports are investigated within 10 business days.

## Conflicts of Interest
Employees must disclose outside employment or financial interests that may conflict with their duties.

## Confidentiality
Employee, customer and company data must be handled according to the data protection policy. Salary data is confidential to HR and Finance.

## Disciplinary Process
Violations follow a progressive process: verbal warning, written warning, final warning, termination. Serious misconduct may result in immediate dismissal.
""",
    "onboarding_and_offboarding.md": """# Onboarding and Offboarding

## Onboarding
New hires complete a 90-day onboarding plan with a buddy, a 30/60/90-day review, and mandatory compliance training within the first 2 weeks. The probation period is 90 days.

## Resignation
Employees must give 30 days' written notice; managers and above give 60 days. Unused annual leave is paid out on the last day.

## Exit Process
HR conducts an exit interview, collects equipment, and revokes system access on the last working day. Alumni are eligible for rehire after 6 months.

## Retention
Departments with annual attrition above 20% must submit a retention action plan to HR each quarter, covering compensation review, career paths and manager feedback.
""",
}


def write_policies(policies_dir: Path) -> None:
    policies_dir.mkdir(parents=True, exist_ok=True)
    for name, text in POLICIES.items():
        (policies_dir / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    from .config import get_settings

    s = get_settings()
    seed_database(Database(s.db_path))
    write_policies(s.policies_dir)
    print(f"Seeded {s.db_path} and wrote policies to {s.policies_dir}")
