"""Read and extract text from PDF"""
import PyPDF2

pdf_path = "d:/epic model/Autonomous Solar Vehicle proposal.pdf"

try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        print(f"Total pages: {num_pages}")
        print("\n" + "="*80)

        full_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            full_text += f"\n--- PAGE {page_num + 1} ---\n{text}\n"

        # Save to file
        with open("d:/epic model/pdf_content.txt", 'w', encoding='utf-8') as out:
            out.write(full_text)

        print("PDF content extracted and saved to: pdf_content.txt")
        print("\nFirst 2000 characters:")
        print(full_text[:2000])

except Exception as e:
    print(f"Error: {e}")
