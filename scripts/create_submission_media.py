from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_assets"
PDF = OUT / "budgetbrain_pitch_deck.pdf"
COVER = OUT / "cover.png"
SCRIPT = OUT / "video_presentation_script.txt"

PAGE_W, PAGE_H = 1280, 720
MARGIN_X = 72
MARGIN_TOP = 58
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#5B6472")
RED = colors.HexColor("#E31B2F")
PANEL = colors.HexColor("#F3F5F8")
LINE = colors.HexColor("#D6DBE3")
DARK = colors.HexColor("#0B0D10")


def fit_text(c, text, max_width, font="Helvetica-Bold", size=56, min_size=32):
    while size > min_size and stringWidth(text, font, size) > max_width:
        size -= 1
    c.setFont(font, size)
    return size


def draw_wrapped(c, text, x, y, width, size=23, leading=32, color=MUTED, font="Helvetica"):
    c.setFillColor(color)
    c.setFont(font, size)
    words = text.split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and stringWidth(candidate, font, size) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def title(c, text, subtitle=None):
    c.setFillColor(INK)
    fit_text(c, text, PAGE_W - 2 * MARGIN_X, size=54)
    c.drawString(MARGIN_X, PAGE_H - MARGIN_TOP - 8, text)
    c.setStrokeColor(RED)
    c.setLineWidth(5)
    c.line(MARGIN_X, PAGE_H - MARGIN_TOP - 30, MARGIN_X + 110, PAGE_H - MARGIN_TOP - 30)
    if subtitle:
        draw_wrapped(c, subtitle, MARGIN_X, PAGE_H - MARGIN_TOP - 72, PAGE_W - 2 * MARGIN_X, 24, 32)


def footer(c, page):
    c.setFillColor(colors.HexColor("#8A93A3"))
    c.setFont("Helvetica", 13)
    c.drawString(MARGIN_X, 38, "BudgetBrain Track 1 Champion Agent")
    c.drawRightString(PAGE_W - MARGIN_X, 38, str(page))


def panel(c, x, y, w, h, heading, body, accent=RED):
    c.setFillColor(PANEL)
    c.setStrokeColor(LINE)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
    c.setFillColor(accent)
    c.rect(x, y + h - 8, w, 8, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 25)
    c.drawString(x + 24, y + h - 48, heading)
    draw_wrapped(c, body, x + 24, y + h - 88, w - 48, 19, 26)


def bullets(c, items, x, y, width, size=24):
    for item in items:
        c.setFillColor(RED)
        c.circle(x + 8, y + 8, 5, fill=1, stroke=0)
        y = draw_wrapped(c, item, x + 28, y, width - 28, size, size + 10, INK)
        y -= 12
    return y


def create_pdf():
    OUT.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(PDF), pagesize=landscape((PAGE_H, PAGE_W)))

    # 1. Cover
    c.setFillColor(DARK)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    if COVER.exists():
        c.drawImage(str(COVER), 0, 0, width=PAGE_W, height=PAGE_H, preserveAspectRatio=True, anchor="c")
    c.showPage()

    # 2. The challenge
    title(c, "The winning problem is hidden", "Track 1 rewards accurate answers under a strict Docker contract, unknown categories, token pressure, and short runtime limits.")
    bullets(
        c,
        [
            "The official input only provides task_id and prompt, so category must be inferred.",
            "The output must exactly match the required JSON answer contract.",
            "Accuracy must pass the gate, but leaderboard position is also shaped by token usage.",
            "The image must run on linux/amd64, stay small, and keep secrets out of the container.",
        ],
        MARGIN_X,
        450,
        980,
        25,
    )
    footer(c, 2)
    c.showPage()

    # 3. Solution
    title(c, "BudgetBrain uses a hybrid solver", "The system answers easy prompts locally and reserves Fireworks calls for ambiguous cases.")
    panel(c, 72, 330, 340, 190, "Local first", "Math, sentiment, named entities, logic, and code debugging can often be solved without spending any tokens.")
    panel(c, 470, 330, 340, 190, "Router driven", "A lightweight classifier chooses the safest path for each prompt based on wording and structure.")
    panel(c, 868, 330, 340, 190, "Model fallback", "Allowed Fireworks models are used only when local confidence is not enough.")
    panel(c, 270, 102, 740, 140, "Optimization target", "Maximize correct answers while minimizing external calls, latency, and token count.")
    footer(c, 3)
    c.showPage()

    # 4. Architecture
    title(c, "The pipeline is simple and compliant")
    x0, y0 = 88, 360
    stages = [
        ("tasks.json", "Read official input"),
        ("Router", "Infer category"),
        ("Local solvers", "Answer deterministic cases"),
        ("Fireworks", "Fallback only when needed"),
        ("results.json", "Write official output"),
    ]
    box_w, box_h, gap = 190, 116, 40
    for i, (head, body) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        c.setFillColor(PANEL if i != 3 else colors.HexColor("#FFF1F2"))
        c.setStrokeColor(RED if i == 3 else LINE)
        c.roundRect(x, y0, box_w, box_h, 8, fill=1, stroke=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 23)
        c.drawCentredString(x + box_w / 2, y0 + 72, head)
        c.setFont("Helvetica", 16)
        c.setFillColor(MUTED)
        c.drawCentredString(x + box_w / 2, y0 + 38, body)
        if i < len(stages) - 1:
            c.setStrokeColor(RED)
            c.setLineWidth(3)
            c.line(x + box_w + 8, y0 + box_h / 2, x + box_w + gap - 8, y0 + box_h / 2)
    draw_wrapped(
        c,
        "Environment variables are injected at runtime: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS. The image contains no API keys.",
        158,
        230,
        960,
        24,
        34,
        INK,
    )
    footer(c, 4)
    c.showPage()

    # 5. Evidence
    title(c, "Validated before submission", "The build was tested across local, Docker, held-out, stress, and real Fireworks runs.")
    rows = [
        ("Unit tests", "33 / 33 passing"),
        ("Official practice", "8 / 8 local and 8 / 8 real Fireworks"),
        ("Held-out set", "16 / 16 passing"),
        ("Reasoning stress", "8 / 8 passing"),
        ("Real complex stress", "8 / 8, 1497 Fireworks tokens"),
        ("Image size", "45.5 MB linux/amd64 Docker image"),
    ]
    left, top = 150, 490
    for idx, (label, value) in enumerate(rows):
        y = top - idx * 62
        c.setFillColor(PANEL if idx % 2 == 0 else colors.white)
        c.rect(left, y - 24, 980, 48, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(left + 24, y - 8, label)
        c.setFont("Helvetica", 22)
        c.setFillColor(MUTED)
        c.drawRightString(left + 940, y - 8, value)
    footer(c, 5)
    c.showPage()

    # 6. Submission
    title(c, "Ready for the Track 1 evaluator", "The public Docker image is available for the official form and anonymous pull verification has passed.")
    bullets(
        c,
        [
            "Docker image: lebinbin/budgetbrain-track1:amd-act2-20260710",
            "Digest: sha256:bb74ac8bf2d2c089a236f578ef82e10e0a9316430fc8f2293bf23468badfedc6",
            "Runtime contract: read /input/tasks.json and write /output/results.json.",
            "Final objective: clear the accuracy gate, then compete on token efficiency and reliability.",
        ],
        MARGIN_X,
        460,
        1060,
        23,
    )
    c.setFillColor(colors.HexColor("#FFF1F2"))
    c.setStrokeColor(RED)
    c.roundRect(96, 104, 1088, 96, 8, fill=1, stroke=1)
    draw_wrapped(c, "Security note: the container does not include secrets. Fireworks credentials are supplied only at runtime by the evaluator.", 128, 158, 1024, 22, 30, INK)
    footer(c, 6)
    c.showPage()
    c.save()


def create_script():
    text = """Video presentation script - 60 to 75 seconds

Hi, this is BudgetBrain Track 1 Champion Agent, our submission for AMD Developer Hackathon Act II Track 1.

The challenge is simple on the surface but difficult in evaluation: the system receives only task_id and prompt, without an explicit category. It must infer the task type, answer correctly, write the exact results.json format, and stay efficient under Docker, runtime, and token constraints.

BudgetBrain solves this with a hybrid strategy. First, a lightweight router classifies the prompt. Then deterministic local solvers handle categories like math, sentiment, named entities, logic, and code debugging whenever the answer can be produced confidently without model tokens. For harder or ambiguous prompts, the agent uses Fireworks models as a controlled fallback, with allowed-model gating, short timeouts, retries, and output normalization.

This gives us two advantages: high reliability on structured tasks and much lower token usage because simple prompts do not need an external model call.

The final image is public on Docker Hub as lebinbin/budgetbrain-track1:amd-act2-20260710. It follows the official contract, reads /input/tasks.json, writes /output/results.json, runs on linux/amd64, and contains no API keys. We validated it with unit tests, official practice tests, held-out tasks, reasoning stress tests, Docker runs, and real Fireworks calls.

Our goal is to pass the hidden accuracy gate and compete for the top leaderboard position through accuracy, speed, and token efficiency.
"""
    SCRIPT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    create_pdf()
    create_script()
    print(PDF)
    print(SCRIPT)
