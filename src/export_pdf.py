"""PDF copies of the deliverables -> deliverables/*.pdf (easy to open and share; the .md/.pptx stay the source).

- Decks: deliverables/From_Trend_to_Waste_Results.pptx and From_Trend_to_Waste_Review2.pptx -> .pdf with LibreOffice.
- Report and summary: reports/final_report.md and reports/project_summary.md -> HTML -> PDF with headless
  Chrome (LibreOffice as a fallback). The Markdown converter below covers only what these generated files use.
Skips quietly when neither tool is installed.
"""
import html
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .config import REPORTS, ROOT

OUT = ROOT / "deliverables"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
DOCS = [("final_report.md", "From_Trend_to_Waste_Report.pdf"), ("project_summary.md", "Project_Summary.pdf"),
        ("review2_guide.md", "Review2_Presenter_Guide.pdf")]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Calibri, "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #3A3532; }
h1, h2, h3, h4 { font-family: Cambria, Georgia, serif; color: #2B2B2B; line-height: 1.2; page-break-after: avoid; }
h1 { font-size: 22pt; margin: 0 0 4pt; } h2 { font-size: 15pt; color: #B5532C; margin-top: 18pt; border-bottom: 1px solid #D8D2C8; padding-bottom: 3pt; }
h3 { font-size: 12.5pt; margin-top: 14pt; } h4 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th { background: #2B2B2B; color: #fff; text-align: left; padding: 4pt 5pt; }
td { border: 1px solid #D8D2C8; padding: 3pt 5pt; vertical-align: top; }
img { max-width: 100%; display: block; margin: 6pt auto 10pt; page-break-inside: avoid; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #F1DED2; border-left: 3px solid #B5532C; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.5pt; background: #F1EEE9; padding: 0 2pt; }
pre { background: #F1EEE9; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; }
hr { border: 0; border-top: 1px solid #D8D2C8; margin: 12pt 0; }
li { margin-bottom: 2pt; }
"""


def inline(text, base):
    text = html.escape(text, quote=False)
    codes = []
    text = re.sub(r"`([^`]+)`", lambda m: codes.append(m.group(1)) or f"\x00{len(codes) - 1}\x00", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)",
                  lambda m: f'<img alt="{m.group(1)}" src="{(base / m.group(2)).resolve().as_uri()}">', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", text)


def md_to_html(md, base):
    out, para, lists, table, quote = [], [], [], [], []
    lines = md.split("\n")

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para), base)}</p>")
            para.clear()
        while lists:
            out.append(f"</li></{lists.pop()[0]}>")
        if table:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not re.match(r"^\|[\s|:-]+\|$", r.strip())]
            out.append("<table><tr>" + "".join(f"<th>{inline(c, base)}</th>" for c in rows[0]) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c, base)}</td>" for c in r) + "</tr>" for r in rows[1:])
                       + "</table>")
            table.clear()
        if quote:
            out.append(f"<blockquote>{inline(' '.join(quote), base)}</blockquote>")
            quote.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        m_h = re.match(r"^(#{1,6})\s+(.*)", line)
        m_li = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
        if not line.strip():
            flush()
        elif m_h:
            flush()
            n = len(m_h.group(1))
            out.append(f"<h{n}>{inline(m_h.group(2), base)}</h{n}>")
        elif re.match(r"^-{3,}\s*$", line):
            flush()
            out.append("<hr>")
        elif line.lstrip().startswith("|"):
            if para or lists or quote:
                flush()
            table.append(line)
        elif line.startswith(">"):
            quote.append(line.lstrip(">").strip())
        elif m_li:
            if para:
                out.append(f"<p>{inline(' '.join(para), base)}</p>")
                para.clear()
            indent, kind = len(m_li.group(1)), ("ol" if m_li.group(2)[0].isdigit() else "ul")
            while lists and lists[-1][1] > indent:
                out.append(f"</li></{lists.pop()[0]}>")
            if lists and lists[-1][1] == indent:
                out.append("</li>")
            else:
                lists.append((kind, indent))
                out.append(f"<{kind}>")
            out.append(f"<li>{inline(m_li.group(3), base)}")
        elif lists:  # continuation of a list item
            out.append(" " + inline(line.strip(), base))
        else:
            para.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)


def to_pdf_chrome(html_path, pdf_path, tmp):
    # headless Chrome sometimes keeps running after printing, so wait for a finished PDF, then stop it
    pdf_path.unlink(missing_ok=True)
    proc = subprocess.Popen([str(CHROME), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                             "--allow-file-access-from-files", "--no-first-run", "--no-default-browser-check",
                             f"--user-data-dir={tmp / 'chrome'}", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last, stable = -1, 0
    for _ in range(240):  # up to 2 minutes
        if proc.poll() is not None:
            break
        size = pdf_path.stat().st_size if pdf_path.exists() else -1
        stable = stable + 1 if size > 0 and size == last else 0
        if stable >= 4:  # unchanged for 2 s
            break
        last = size
        time.sleep(0.5)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Chrome did not write {pdf_path}")


def to_pdf_soffice(src, outdir):
    subprocess.run([shutil.which("soffice"), "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(src)],
                   check=True, capture_output=True, timeout=300)


def main():
    soffice = shutil.which("soffice")
    for deck in (OUT / "From_Trend_to_Waste_Results.pptx", OUT / "From_Trend_to_Waste_Review2.pptx"):
        if soffice and deck.exists():
            to_pdf_soffice(deck, OUT)
            print(f"wrote {OUT / deck.with_suffix('.pdf').name}")
        elif not soffice:
            print("LibreOffice (soffice) not found: skip deck PDF")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for src, target in DOCS:
            if not (REPORTS / src).exists():
                print(f"reports/{src} not found: skip {target}")
                continue
            md = (REPORTS / src).read_text()
            title = re.search(r"^#\s+(.*)", md, re.M).group(1)
            page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                    f"<style>{CSS}</style></head><body>{md_to_html(md, REPORTS)}</body></html>")
            h = tmp / (Path(target).stem + ".html")
            h.write_text(page)
            if CHROME.exists():
                to_pdf_chrome(h, OUT / target, tmp)
            elif soffice:
                to_pdf_soffice(h, tmp)
                shutil.move(tmp / (h.stem + ".pdf"), OUT / target)
            else:
                print(f"no PDF tool found: skip {target}")
                continue
            print(f"wrote {OUT / target}")


if __name__ == "__main__":
    main()
