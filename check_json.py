import json

input_file = '/data/audiocaps/test_audiocaps_subset_updated.json'
output_file = '/data/audiocaps/formatted_audiocaps.json'

def convert_jsonl_to_array(input_path, output_path):
    result_array = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Parse each line as an individual JSON object
                data = json.loads(line)
                
                # Reorder and filter keys
                formatted_entry = {
                    "captions": data.get("captions"),
                    "location": data.get("location")
                }
                result_array.append(formatted_entry)
    
    # Save the final list as a pretty-printed JSON array
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_array, f, indent=4)

if __name__ == "__main__":
    convert_jsonl_to_array(input_file, output_file)
    print(f"Success! Final array saved to {output_file}")