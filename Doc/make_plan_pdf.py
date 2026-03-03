#!/usr/bin/env python3
"""Generate PDF from project_plan.md. Edit the .md file and run this script to update the PDF."""
import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate


def md_to_reportlab(md_text: str) -> str:
    """Convert markdown to ReportLab paragraph markup (bold, bullets, line breaks)."""
    out: list[str] = []
    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("<br/>")
            continue
        # Bold: **text** -> <b>text</b>
        stripped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", stripped)
        # Inline code: `text` -> <font face='Courier' size='8'>text</font>
        stripped = re.sub(r"`([^`]+)`", r"<font face='Courier' size='8'>\1</font>", stripped)
        if stripped.startswith("# "):
            out.append(f"<b>{stripped[2:]}</b><br/><br/>")
        elif stripped.startswith("## "):
            out.append(f"<b>{stripped[3:]}</b><br/>")
        elif stripped.startswith("- "):
            out.append(f"• {stripped[2:]}<br/>")
        else:
            out.append(f"{stripped}<br/>")
    return "".join(out)


def main():
    script_dir = Path(__file__).resolve().parent
    md_path = script_dir / "project_plan.md"
    out_path = script_dir / "group_project_plan.pdf"

    if not md_path.exists():
        raise SystemExit(f"Markdown file not found: {md_path}")

    content = md_path.read_text(encoding="utf-8")
    body = md_to_reportlab(content)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        "Plan",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        spaceAfter=0,
    )
    story = [Paragraph(body, style)]
    doc.build(story)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
