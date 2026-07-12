from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_assets"
PDF = OUT / "budgetbrain_pitch_deck.pdf"
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

LEADERBOARD_BASELINE_COMMIT = "1393b47"
LEADERBOARD_RANK = "34"
LEADERBOARD_ACCURACY = "89.5%"
LEADERBOARD_TOKENS = "5,093"
CURRENT_RELEASE_COMMIT = "Pending until commit"
UNIT_TESTS = "79 / 79 passing"
OFFLINE_STRICT = "10 / 10 passing"
OFFLINE_AUDIT = "19 / 19 passing"
DOCKER_TAG = "Pending / to be filled"
DOCKER_DIGEST = "Pending / to be filled"


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
    rl_config.useA85 = 0
    c = canvas.Canvas(str(PDF), pagesize=landscape((PAGE_H, PAGE_W)), pageCompression=1)

    # 1. Cover
    c.setFillColor(DARK)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 66)
    c.drawString(92, 520, "BudgetBrain")
    c.setFillColor(RED)
    c.drawString(92, 438, "Track 1 Champion Agent")
    c.setFillColor(colors.HexColor("#D1D5DB"))
    c.setFont("Helvetica", 28)
    c.drawString(96, 354, "Accurate answers. Controlled tokens. Docker-ready contract.")
    c.setStrokeColor(RED)
    c.setLineWidth(5)
    c.line(96, 314, 1184, 314)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(96, 250, "OFFICIAL LEADERBOARD BASELINE")
    c.setFont("Helvetica", 22)
    c.drawString(96, 212, "Commit 1393b47  |  Rank 34  |  89.5% accuracy  |  5,093 tokens")
    c.setFont("Helvetica-Bold", 22)
    c.drawString(96, 142, "CURRENT RELEASE VALIDATION")
    c.setFont("Helvetica", 22)
    c.drawString(96, 104, "Commit pending  |  79/79 tests  |  strict 10/10  |  audit 19/19")
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
    title(c, "Evidence snapshot", "Official leaderboard evidence is reported separately from offline regression fixtures.")
    rows = [
        ("Leaderboard baseline commit", LEADERBOARD_BASELINE_COMMIT),
        ("Official leaderboard", f"Rank {LEADERBOARD_RANK}"),
        ("Official hidden result", f"{LEADERBOARD_ACCURACY}, {LEADERBOARD_TOKENS} tokens"),
        ("Current release commit", CURRENT_RELEASE_COMMIT),
        ("Unit tests", UNIT_TESTS),
        ("Offline validation", f"strict {OFFLINE_STRICT}; audit {OFFLINE_AUDIT}"),
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
    title(c, "Release artifact pending", "Do not attach a historical Docker tag or digest to the current source snapshot.")
    bullets(
        c,
        [
            f"Docker image: {DOCKER_TAG}",
            f"Digest: {DOCKER_DIGEST}",
            "Runtime contract: read /input/tasks.json and write /output/results.json.",
            "Offline fixtures are regression checks, not official hidden-evaluator results.",
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
    text = f"""Video presentation script - 60 to 75 seconds

Hi, this is BudgetBrain Track 1 Champion Agent, our submission for AMD Developer Hackathon ACT II Track 1.

Track 1 provides only a task ID and prompt. The agent must infer the task type, return the exact results JSON contract, stay reliable in a Linux AMD64 container, and control model-token usage without sacrificing correctness.

BudgetBrain uses a hybrid strategy. Narrow deterministic solvers handle tasks only when their outputs can be verified. Harder or ambiguous prompts use runtime-allowed Fireworks models with output validation and fallback handling. Accuracy remains the first priority, and token reduction is applied only to proven local paths.

The official leaderboard baseline is commit {LEADERBOARD_BASELINE_COMMIT}: rank {LEADERBOARD_RANK}, {LEADERBOARD_ACCURACY} hidden accuracy, and {LEADERBOARD_TOKENS} tokens. That score does not represent the current uncommitted release worktree. Current release validation passes 79 of 79 unit tests, a 10-of-10 offline strict fixture, and a 19-of-19 offline audit. Those offline fixtures validate regressions and are not claims about hidden accuracy.

The Track 1 container reads /input/tasks.json and writes /output/results.json. Its Fireworks credentials and allowed models are injected only at runtime. The current release commit, Docker tag, and digest are pending final commit, build, publication, and anonymous verification.
"""
    SCRIPT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    create_pdf()
    create_script()
    print(PDF)
    print(SCRIPT)
