UNIFIED_CODE_COLUMNS = {
    "source": str,         # dataset name (e.g., 'LeetCodeDataset')
    "dataset_id": str,     # id inside the dataset if available
    "title": str,          # problem title
    "prompt": str,         # problem/question text
    "solution": str,       # reference solution/code if present
    "language": str,       # 'python','cpp', etc.
    "difficulty": str,     # 'easy','medium','hard' if present
    "tags": str,           # comma-joined tags
    "split": str           # 'train','test','validation' if applicable
}
