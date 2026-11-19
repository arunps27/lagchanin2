"""Seed the SQLite bank DB (read via the MCP sqlite server) and generate sample KYC PDFs."""
import sqlite3, os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "bank.db"
DOCS = ROOT / "data" / "docs"

CUSTOMERS = [
 # id, name, dob, address, phone, employer, stated_income, id_number, id_expiry
 (1,"Priya Sharma","1988-04-12","12 Lotus Lane, Pune","+91-9800011111","Infosys Ltd",1800000,"ABCPS1234K","2031-05-01"),
 (2,"Vladimir Petrov","1975-09-03","77 Harbor Road, Mumbai","+91-9800022222","Petrov Trading LLC",2500000,"PQRVP5678L","2029-01-15"),
 (3,"Rahul Verma","1990-01-22","5 Orchid Street, Delhi","+91-9800033333","Self-employed",900000,"LMNRV9012M","2023-02-01"),
 (4,"Anita Desai","1985-07-30","5 Orchid Street, Delhi","+91-9800033333","Quick Cash Traders",4000000,"XYZAD3456N","2030-08-20"),
 (5,"Karan Mehta","1993-11-05","5 Orchid Street, Delhi","+91-9800044444","Quick Cash Traders",3800000,"DEFKM7890P","2032-03-10"),
 (6,"Meera Nair","1992-02-14","90 Palm Grove, Kochi","+91-9800055555","TCS",1500000,"GHIMN1122Q","2030-12-31"),
]
APPS = [
 # app_id, customer_id, amount, purpose, tenure_months, declared_monthly_obligations
 ("APP-1001",1,2500000,"Home renovation",60,20000),
 ("APP-1002",2,9000000,"Business expansion",84,50000),
 ("APP-1003",3,1500000,"Personal loan",48,45000),
 ("APP-1004",4,6000000,"Working capital",36,30000),
 ("APP-1005",6,800000,"Education",48,10000),
]
ACCOUNTS = [(1,1,"HDFC",620000,"2019-03-01"),(2,2,"Offshore-Cayman Bank",12000000,"2024-09-10"),
            (3,3,"SBI",40000,"2023-06-01"),(4,4,"Axis",3500000,"2024-10-01"),(5,5,"Axis",3300000,"2024-10-02"),(6,6,"ICICI",410000,"2020-01-01")]
WATCHLIST = [
 ("W-01","Vladimir Petrov","1975-09-03","OFAC SDN","Sanctioned individual - energy sector"),
 ("W-02","Quick Cash Traders","", "FIU-IND Alert","Entity flagged for suspected layering / money-mule activity"),
 ("W-03","Ivan Kuznetsov","1969-01-01","EU Sanctions","Asset freeze"),
]
POLICY = [
 ("P-1","KYC","Government ID must be valid (not expired). Expired ID => KYC FAIL until refreshed."),
 ("P-2","Sanctions","Any exact or near match to a sanctions/watchlist entity => REJECT and escalate to compliance (no loan)."),
 ("P-3","Fraud ring","Multiple unrelated applicants sharing the same address/phone or employer flagged by FIU => MANUAL REVIEW by fraud team."),
 ("P-4","Affordability","Debt-to-income: (declared obligations + new EMI) / monthly income must be <= 50% else REJECT or reduce amount."),
 ("P-5","Income verification","Stated income must be within 20% of income on the submitted payslip/KYC document else MANUAL REVIEW."),
 ("P-6","Auto-approve","KYC PASS, no watchlist hits, no ring flags, DTI <= 40% => AUTO APPROVE."),
]

def build():
    DB.parent.mkdir(exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    if DB.exists(): DB.unlink()
    c = sqlite3.connect(DB)
    c.executescript("""
    CREATE TABLE customers(id INTEGER PRIMARY KEY,name TEXT,dob TEXT,address TEXT,phone TEXT,employer TEXT,stated_income INTEGER,id_number TEXT,id_expiry TEXT);
    CREATE TABLE applications(app_id TEXT PRIMARY KEY,customer_id INTEGER,amount INTEGER,purpose TEXT,tenure_months INTEGER,monthly_obligations INTEGER);
    CREATE TABLE accounts(id INTEGER PRIMARY KEY,customer_id INTEGER,bank TEXT,balance INTEGER,opened TEXT);
    CREATE TABLE watchlist(wid TEXT PRIMARY KEY,name TEXT,dob TEXT,source TEXT,note TEXT);
    CREATE TABLE policy(pid TEXT PRIMARY KEY,topic TEXT,rule TEXT);
    """)
    c.executemany("INSERT INTO customers VALUES(?,?,?,?,?,?,?,?,?)",CUSTOMERS)
    c.executemany("INSERT INTO applications VALUES(?,?,?,?,?,?)",APPS)
    c.executemany("INSERT INTO accounts VALUES(?,?,?,?,?)",ACCOUNTS)
    c.executemany("INSERT INTO watchlist VALUES(?,?,?,?,?)",WATCHLIST)
    c.executemany("INSERT INTO policy VALUES(?,?,?)",POLICY)
    c.commit(); c.close()
    # KYC PDFs: payslip income deliberately differs for APP-1004 (income mismatch)
    payslip_annual = {1:1800000,2:2500000,3:900000,4:1200000,6:1500000}
    for app_id,cid,*_ in APPS:
        cu = next(x for x in CUSTOMERS if x[0]==cid)
        p = canvas.Canvas(str(DOCS/f"{app_id}_kyc.pdf"),pagesize=A4)
        y=800
        for line in ["KYC & INCOME DOCUMENT PACK",f"Application: {app_id}",f"Full Name: {cu[1]}",f"Date of Birth: {cu[2]}",
                     f"Address: {cu[3]}",f"Phone: {cu[4]}",f"Government ID: {cu[7]}  (expires {cu[8]})",
                     f"Employer: {cu[5]}",f"Annual income per payslip/ITR: INR {payslip_annual[cid]}"]:
            p.drawString(60,y,line); y-=24
        p.save()

if __name__=="__main__":
    build(); print("seeded",DB)
