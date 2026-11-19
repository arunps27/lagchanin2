"""Writes a small text-based PDF (no extra deps) for testing PDF upload."""
lines = [
 "Employee Onboarding Handbook",
 "",
 "1. Laptop and equipment. New hires receive a managed laptop on day one.",
 "Devices must be returned within 10 business days of leaving the company.",
 "",
 "2. Mandatory training. All employees must complete the security awareness",
 "course within 14 days of joining. The course is renewed every 12 months.",
 "",
 "3. Probation period. The probation period lasts 90 days, with a review",
 "meeting at day 45 and a final review at day 90 with the line manager.",
 "",
 "4. Working hours. Core hours are 10:00 to 16:00 local time. Employees",
 "may work remotely up to 3 days per week with manager approval.",
 "",
 "5. Mentorship. Each new hire is paired with a buddy for the first 60 days.",
 "Buddies meet the new hire weekly to answer questions about tools and process.",
]
def esc(s): return s.replace("\\","\\\\").replace("(","\(").replace(")","\)")
content = "BT /F1 12 Tf 50 780 Td 16 TL\n" + "\n".join(f"({esc(l)}) '" for l in lines) + "\nET"
objs = [
 "<< /Type /Catalog /Pages 2 0 R >>",
 "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
 "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
 f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
 "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
]
out = "%PDF-1.4\n"; offs = []
for i, o in enumerate(objs, 1):
    offs.append(len(out.encode("latin-1"))); out += f"{i} 0 obj\n{o}\nendobj\n"
x = len(out.encode("latin-1"))
out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offs)
out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n"
open("samples/onboarding_handbook.pdf", "wb").write(out.encode("latin-1"))
