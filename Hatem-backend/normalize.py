import re

def remove_diacritics(text):
    """Remove Arabic diacritics (Tashkeel)."""
    arabic_diacritics = re.compile(r'[\u064B-\u065F]')
    return arabic_diacritics.sub('', text)

def normalize_arabic(text):
    """Normalize Arabic letters for consistency."""
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")  # Normalize Alef variations
    text = text.replace("ة", "ه")  # Convert Ta Marbuta to Ha
    return text

def normalize_text_file(input_file, output_file):
    """Reads a text file, normalizes its content, and saves the output."""
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Apply normalization
        normalized_text = remove_diacritics(text)
        normalized_text = normalize_arabic(normalized_text)
        
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(normalized_text)
        
        print(f"✅ Normalization complete! Saved as: {output_file}")
    
    except Exception as e:
        print(f"⚠️ Error: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "studies_and_exams.txt"  # Replace with your actual file name
    output_file = f"norm_{input_file}"
    
    normalize_text_file(input_file, output_file)
