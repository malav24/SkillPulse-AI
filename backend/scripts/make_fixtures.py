"""Generate SYNTHETIC test fixtures (clearly labelled TEST DATA — not a real university syllabus).

Usage:  python scripts/make_fixtures.py [output_dir]
Creates:
  sample_syllabus.pdf   3-page selectable-text syllabus (header, objectives, outcomes, units, labs, books)
  scanned_syllabus.pdf  same kind of content rendered as an image only (exercises OCR fallback)
  blank.pdf             a single empty page (insufficient text)
  corrupted.pdf         starts with %PDF- but is truncated garbage
  not_a_pdf.pdf         plain text with a .pdf extension
  sample_import.csv     university/programme metadata CSV with valid and invalid rows
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BANNER = "SYNTHETIC TEST DATA - generated for SkillPulse AI automated tests"

PAGES = [
    [
        BANNER,
        "Example Technological University",
        "Faculty of Engineering",
        "Programme: B.Tech Computer Science and Engineering",
        "Course Code: CS501",
        "Course Title: Machine Learning and Data Analytics",
        "Semester: V        Credits: 4",
        "Academic Year: 2024-25",
        "",
        "Course Objectives",
        "1. To introduce the fundamentals of ML and supervised learning.",
        "2. To develop data analysis skills using Python and pandas.",
        "3. To build awareness of professional ethics in AI systems.",
        "",
        "Course Outcomes",
        "After completion of the course, students will be able to:",
        "CO1: Explain core concepts of machine learning and probability.",
        "CO2: Apply regression and classification algorithms using scikit-learn.",
        "CO3: Design and evaluate neural networks with TensorFlow.",
        "CO4: Communicate results through data visualization and technical writing.",
    ],
    [
        "Unit I: Introduction to Machine Learning (8 Hours)",
        "Types of learning; supervised learning and unsupervised learning. Bias-variance trade-off,",
        "cross-validation, overfitting. Linear algebra and statistics refresher.",
        "",
        "Unit II: Data Preprocessing and SQL (8 Hours)",
        "Data cleaning, feature engineering, handling missing values. Querying data with SQL and MySQL.",
        "Exploratory data analysis with pandas and NumPy.",
        "",
        "Unit III: Supervised Algorithms",
        "Decision trees, random forests, support vector machine, k-nearest neighbours.",
        "Model evaluation: confusion matrix, precision and recall.",
        "",
        "UNIT - IV",
        "Neural Networks and Deep Learning",
        "Perceptron, backpropagation, convolutional neural network (CNN) for image recognition.",
        "Natural Language Processing basics: tokenisation and text classification with NLP pipelines.",
        "",
        "Unit V: Deployment (6 Hours)",
        "Deploying models as REST APIs with Flask; containers with Docker; using AWS cloud services",
        "such as EC2 and S3. Version control with Git.",
    ],
    [
        "List of Practicals",
        "1. Implement linear regression in Python using NumPy.",
        "2. Perform exploratory data analysis on a CSV dataset with pandas and Matplotlib.",
        "3. Train a CNN on an image dataset using Keras.",
        "4. Build a REST API for a trained model and containerise it with Docker.",
        "",
        "Mini Project",
        "Students work in teams to build an end-to-end data analytics project and present it.",
        "",
        "Text Books",
        "1. Introduction to Machine Learning, Example Author, Example Press, 2020.",
        "2. Python for Data Analysis, Example Author, Example Publisher, 2022.",
        "",
        "Reference Books",
        "1. Pattern Recognition and Machine Learning, Example Author, 2019.",
        "",
        "The term AWS here is unrelated: AWS = Assignment Writing Schedule (ambiguity test).",
        "Students may excel in R and C when supported by tutors.",
    ],
]


def _text_pdf(pages: list[list[str]]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle("SkillPulse synthetic syllabus (TEST DATA)")
    for lines in pages:
        y = 800
        for ln in lines:
            c.setFont("Helvetica-Bold" if ln and (ln.startswith(("Unit", "UNIT", "Course O", "List", "Text",
                                                                 "Reference", "Mini"))) else "Helvetica", 10)
            c.drawString(50, y, ln)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def _scanned_pdf(pages: list[list[str]]) -> bytes:
    """Render text into images and embed only the images (no text layer)."""
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    out = fitz.open()
    for lines in pages:
        img = Image.new("L", (1240, 1754), 255)  # A4 @ 150 dpi
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        y = 120
        for ln in lines:
            draw.text((80, y), ln, fill=0, font=font)
            y += 40
        png = io.BytesIO()
        img.convert("1").save(png, format="PNG", optimize=True)
        page = out.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=png.getvalue())
    data = out.tobytes()
    out.close()
    return data


def _blank_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.showPage()
    c.save()
    return buf.getvalue()


CSV = """university_name,university_type,state,district,official_website,source_url,verified_identifier,identifier_source,programme_name,degree,branch,specialization,academic_level,institution,duration_years
Example Technological University (TEST DATA),state,Gujarat,Vadodara,https://example.edu.in,https://example.edu.in/about,,,B.Tech Computer Science,B.Tech,Computer Science and Engineering,,UG,,4
Example Technological University (TEST DATA),state,Gujarat,Vadodara,https://example.edu.in,,,,M.Tech Data Science,M.Tech,Computer Science,Data Science,PG,,2
Sample Institute of Science (TEST DATA),deemed,Karnataka,Bengaluru,https://sample-institute.ac.in,,,,MCA,MCA,Computer Applications,,postgraduate,,2
Invalid State University (TEST DATA),state,Atlantis,,https://invalid.example.in,,,,,,,,,,
Bad Website University (TEST DATA),private,Kerala,,not-a-url,,,,BBA,BBA,Management,,undergraduate,,3
,state,Gujarat,,,,,,,,,,,,
Example Technological University (TEST DATA),state,Gujarat,Vadodara,https://example.edu.in,https://example.edu.in/about,,,B.Tech Computer Science,B.Tech,Computer Science and Engineering,,UG,,4
Bad Level University (TEST DATA),central,Delhi,,,,,,PhD CS,PhD,CS,,postdoc,,
"""


def build(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sample_syllabus.pdf": _text_pdf(PAGES),
        "scanned_syllabus.pdf": _scanned_pdf([PAGES[0][1:12], PAGES[1][:8]]),
        "blank.pdf": _blank_pdf(),
        "corrupted.pdf": b"%PDF-1.7\n1 0 obj << /Type /Catalog /Pages 2 0 R >>\nthis is truncated garbage \x00\x01",
        "not_a_pdf.pdf": b"This is a plain text file pretending to be a PDF.\n",
        "sample_import.csv": CSV.encode(),
    }
    paths = {}
    for name, data in files.items():
        p = out_dir / name
        p.write_bytes(data)
        paths[name] = p
    return paths


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    for name, p in build(target).items():
        print(f"wrote {p} ({p.stat().st_size} bytes)")
