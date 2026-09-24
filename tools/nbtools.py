"""Small helpers to write the session notebooks with nbformat.

Conventions
- Section numbers (e.g. 1.3) match the section numbers used on the slides.
- Exercise A = run & tweak (change a parameter, interpret). Exercise B = optional, write a few lines.
- Solutions are Colab form cells: the code is hidden until you click "Show code".
- GITHUB_REPO = "OWNER/REPO" is a placeholder; tools/set_repo_links.py replaces it everywhere.
"""
import nbformat as nbf

GITHUB_REPO = "jochendeweerdt/Session-day-1---Data-preprocessing"
BRANCH = "main"


def colab_url(nb_file):
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{BRANCH}/notebooks/{nb_file}"


class NB:
    def __init__(self, filename):
        self.filename = filename
        self.nb = nbf.v4.new_notebook()
        self.nb.metadata = {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        }

    def md(self, text):
        self.nb.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, src):
        self.nb.cells.append(nbf.v4.new_code_cell(src.strip("\n")))

    def solution(self, title, src):
        cell = nbf.v4.new_code_cell(f"#@title {title}\n" + src.strip("\n"))
        cell.metadata = {"cellView": "form"}
        self.nb.cells.append(cell)

    def exercise(self, label, text):
        self.md(f"### Exercise {label}\n\n" + text.strip("\n"))

    def header(self, title, section, goal, duration):
        self.md(f"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(self.filename)})

# {title}

**Postgraduate Studies in Business Analytics & AI – Session day 1** · Prof. dr. Jochen De Weerdt (KU Leuven)
Slides: *Session 1.2 – Data preprocessing, EDA and anomaly detection*, part {section}

**Goal.** {goal}

**Time in class:** about {duration} minutes.

**How to use this notebook**
1. `File → Save a copy in Drive` first, so your changes are kept.
2. Run the cells from top to bottom (`Shift + Enter`). The lecturer demo sections run as they are.
3. **Exercise A** (everyone): change a parameter or a line that is marked `# <- change`, rerun, and answer the question.
4. **Exercise B** (optional): write a few lines of code yourself.
5. Solutions are in hidden cells titled *Solution*. Click *Show code* only after you tried.

Section numbers (e.g. {section}.3) correspond to the section numbers on the slides.
""")

    def data_cell(self, extra_install=""):
        install = f"%pip install -q {extra_install}\n" if extra_install else ""
        self.code(f"""
{install}import os
# Data is read from the course GitHub repository (or from ../data when you run the notebook locally)
GITHUB_REPO = "{GITHUB_REPO}"
DATA = "../data/" if os.path.exists("../data") else f"https://raw.githubusercontent.com/{{GITHUB_REPO}}/{BRANCH}/data/"
print("Reading data from:", DATA)
""")

    def save(self, folder):
        nbf.write(self.nb, f"{folder}/{self.filename}")
