import pymupdf as fitz  # This eliminates the warning

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Read PDF
pdf_path = "data/Placement Policy 2027_B.Tech_MCA.pdf"

document = fitz.open(pdf_path)

text = ""

for page_number, page in enumerate(document):

    page_text = page.get_text()

    text += f"\n[Page {page_number + 1}]\n"
    text += page_text

document.close()


# Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)


print("Total chunks:", len(chunks))


# Display first 3 chunks
for i, chunk in enumerate(chunks[:3]):

    print("\n==============================")
    print("CHUNK:", i + 1)
    print("==============================")

    print(chunk)