import json
import re

def parse_entry(entry: str) -> dict:
    """Parse a single entry to extract angry and neutral sentences."""
    # Remove quotes if present at the start/end
    entry = entry.strip()
    if entry.startswith('"') and entry.endswith('"'):
        entry = entry[1:-1]
    
    # Try to find ANGER: and NEUTRAL: markers
    anger_match = re.search(r'ANGER:\s*(.+?)(?:\s*NEUTRAL:|$)', entry, re.DOTALL | re.IGNORECASE)
    neutral_match = re.search(r'NEUTRAL:\s*(.+?)$', entry, re.DOTALL | re.IGNORECASE)
    
    if anger_match and neutral_match:
        # Both markers found
        angry = anger_match.group(1).strip()
        neutral = neutral_match.group(1).strip()
    else:
        # No markers found, try to split by common patterns
        # Look for sentence boundaries (period followed by space and capital letter)
        # or newline separators
        if '\n' in entry:
            parts = entry.split('\n', 1)
            if len(parts) == 2:
                angry = parts[0].strip()
                neutral = parts[1].strip()
            else:
                # Fallback: split by period and space followed by capital - keep period with first sentence
                match = re.search(r'^(.+?[.!?])\s+([A-Z].+)$', entry)
                if match:
                    angry = match.group(1).strip()
                    neutral = match.group(2).strip()
                else:
                    # Last resort: split in half
                    mid = len(entry) // 2
                    angry = entry[:mid].strip()
                    neutral = entry[mid:].strip()
        else:
            # Try to find sentence boundary - keep period with first sentence
            match = re.search(r'^(.+?[.!?])\s+([A-Z].+)$', entry)
            if match:
                angry = match.group(1).strip()
                neutral = match.group(2).strip()
            else:
                # Fallback: split by exclamation mark or question mark
                match = re.search(r'^(.+?[!?])\s+(.+)$', entry)
                if match:
                    angry = match.group(1).strip()
                    neutral = match.group(2).strip()
                else:
                    # Last resort: split in half
                    mid = len(entry) // 2
                    angry = entry[:mid].strip()
                    neutral = entry[mid:].strip()
    
    # Clean up quotes if present
    if angry.startswith('"') and angry.endswith('"'):
        angry = angry[1:-1]
    if neutral.startswith('"') and neutral.endswith('"'):
        neutral = neutral[1:-1]
    
    return {"angry": angry, "neutral": neutral}

def transform_json_file(input_file: str, output_file: str = None):
    """Transform a JSON file from list of strings to list of dicts."""
    if output_file is None:
        output_file = input_file
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    transformed = [parse_entry(entry) for entry in data]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)
    
    print(f"Transformed {len(transformed)} entries in {input_file}")

if __name__ == "__main__":
    transform_json_file("data/train.json")
    transform_json_file("data/test.json")

