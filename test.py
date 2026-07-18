import fitz

doc = fitz.open("data\raw\Second-world-war.pdf")

page = doc[0]

blocks = page.get_text("dict")["blocks"]

for block in blocks:
    if "lines" not in block:
        continue

    for line in block["lines"]:
        for span in line["spans"]:
            print(
                span["text"],
                span["size"],
                span["font"]
            )