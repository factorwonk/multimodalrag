import json
import pandas as pd
import numpy as np

def convfinqadfloader(filepath: str, max_rows: int=1000) -> pd.DataFrame:
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if max_rows is not None:
        data = data[:max_rows]

    # Initialize list to store flattened data
    flattened_data = []
    
    # Process each record in the JSON data
    for item in data:
        # Create base item with common fields
        base_item = {
            'id': item.get('id'),
            'pre_text': ' '.join(item.get('pre_text', [])),
            'post_text': ' '.join(item.get('post_text', [])),
            'filename': item.get('filename'),
            'table': str(item.get('table')),
        }

    # handle annotation which can be either a dictionary or a list
        annotation = item.get('annotation', {})
        if isinstance(annotation, dict):
            # Extract dialogue information
            dialogue_break = annotation.get('dialogue_break', [])
            turn_program = annotation.get('turn_program', [])
            qa_split = annotation.get('qa_split', [])
            exe_ans_list = annotation.get('exe_ans_list', [])
        
        # Create a row for each turn in the dialogue
            for idx in range(len(dialogue_break)):
                turn_data = {
                    'dialogue_text': dialogue_break[idx] if idx < len(dialogue_break) else None,
                    'turn_program': turn_program[idx] if idx < len(turn_program) else None,
                    'qa_split': qa_split[idx] if idx < len(qa_split) else None,
                    'execution_answer': exe_ans_list[idx] if idx < len(exe_ans_list) else None,
                    'turn_index': idx
                }
                
                # Combine base item with turn data
                combined_data = {**base_item, **turn_data}
                flattened_data.append(combined_data)
        
        # Handle potential QA pairs stored directly
        if 'qa' in item:
            qa_data = {
                'question': item['qa'].get('question'),
                'answer': item['qa'].get('answer'),
                'explanation': item['qa'].get('explanation'),
                'program': item['qa'].get('program'),
                'execution_answer': item['qa'].get('exe_ans'),
                'turn_index': 0  # Single QA pair
            }
            combined_data = {**base_item, **qa_data}
            flattened_data.append(combined_data)
    
    # Create DataFrame from flattened data
    df = pd.DataFrame(flattened_data)

    # Convert appropriate columns to proper numeric types
    numeric_columns = ['turn_index', 'qa_split', 'execution_answer']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df