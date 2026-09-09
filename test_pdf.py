import pymupdf as fitz  # This eliminates the warning

pdf_path = "data/Placement Policy 2027_B.Tech_MCA.pdf"

document = fitz.open(pdf_path)

print("PDF opened successfully!")
print("Number of pages:", len(document))

for page_number, page in enumerate(document):
    text = page.get_text()

    print("\n==============================")
    print("PAGE:", page_number + 1)
    print("==============================")

    print(text[:500])

document.close()