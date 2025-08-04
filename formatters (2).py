def format_flattened_output(structured_output):
    """Format the flattened output with block type and segments."""
    reformatted_flattened = {}
    for block_id, block_data in structured_output.items():
        # Determine the block type (tag/attr/meta/jsonld)
        block_type = (
            block_data.get("tag") or 
            block_data.get("attr") or 
            block_data.get("meta") or 
            block_data.get("jsonld") or 
            "unknown"
        )
    
        # Get full text (fallback: join all sentences)
        full_text = block_data.get("text", " ".join(
            s_data["text"] for s_data in block_data["tokens"].values()
        ))
    
        reformatted_flattened[block_id] = {
            "type": block_type,  # "p", "alt", "og:title", etc.
            "text": full_text,
            "segments": {  # Renamed from "tokens" for clarity
                f"{block_id}_{s_key}": s_data["text"]
                for s_key, s_data in block_data["tokens"].items()
            }
        }
    
    return reformatted_flattened


def create_categorized_sentences(flattened_output, structured_output):
    """Create categorized sentences structure from flattened output."""
    flat_sentences_only = {
        k: v for k, v in flattened_output.items()
        if "_S" in k and "_W" not in k
    }
    
    # Create categorized structure for flat_sentences_only
    categorized_sentences = {
        "1_word": [],
        "2_words": [],
        "3_words": [],
        "4_or_more_words": []
    }
    
    # Group blocks by text content and tag
    text_tag_groups = {}
    for block_id, text in flat_sentences_only.items():
        # Get block number from block_id
        block_num = block_id.split('_')[1]
        full_block_id = f"BLOCK_{block_num}"
        
        # Get tag information from structured_output
        block_data = structured_output.get(full_block_id, {})
        tag_type = (
            block_data.get("tag") or 
            block_data.get("attr") or 
            block_data.get("meta") or 
            block_data.get("jsonld") or 
            "unknown"
        )
        
        # Create composite key for text and tag combination
        key = f"{text}||{tag_type}"
        
        if key not in text_tag_groups:
            text_tag_groups[key] = {
                "text": text,
                "tag": tag_type,
                "blocks": []
            }
        
        text_tag_groups[key]["blocks"].append(block_id)
    
    # Process groups and categorize by word count
    for combo_data in text_tag_groups.values():
        # Count words in text
        word_count = len(combo_data["text"].split())
        
        # Determine category
        if word_count == 1:
            category = "1_word"
        elif word_count == 2:
            category = "2_words"
        elif word_count == 3:
            category = "3_words"
        else:
            category = "4_or_more_words"
        
        blocks = combo_data["blocks"]
        
        # For 1-3 word entries with the same text and tag, merge them
        if category != "4_or_more_words" and len(blocks) > 1:
            # Create a merged block ID key
            merged_block_id = "=".join(blocks)
            
            # Create the entry with proper JSON structure
            entry = {
                merged_block_id: combo_data["text"],
                "tag": f"<{combo_data['tag']}>"
            }
            categorized_sentences[category].append(entry)
        else:
            # For 4+ words or unique entries, add individual entries
            for block_id in blocks:
                entry = {
                    block_id: combo_data["text"],
                    "tag": f"<{combo_data['tag']}>"
                }
                categorized_sentences[category].append(entry)
    
    return categorized_sentences