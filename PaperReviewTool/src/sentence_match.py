"""
 * @file Extract matching text segments from context based on fuzzy matching 
 * @author Zewei Ma <zewei.ma@outlook.com>, developed with the help of Claude
 * @copyright Zewei Ma, UIUC
 */
"""

import re
from rapidfuzz import fuzz

def clean_for_matching(text):
    """Clean text for fuzzy matching while preserving structure"""
    # Remove LaTeX commands: \command, \command{}, etc.
    text = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
    # Remove markdown table delimiters but keep content
    text = re.sub(r'\|', ' ', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def render_latex_for_display(text):
    """Convert LaTeX to readable format for display"""
    # Convert common LaTeX commands to readable text
    text = re.sub(r'\$\\mathrm\{([^}]+)\}\$', r'\1', text)
    text = re.sub(r'\$\\mathsf\{([^}]+)\}\$', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathsf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    
    # Handle subscripts and superscripts
    text = re.sub(r'_\{([^}]+)\}', r'_\1', text)
    text = re.sub(r'\^\{([^}]+)\}', r'^(\1)', text)
    
    # Remove dollar signs for inline math
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    
    # Clean up remaining backslashes
    text = re.sub(r'\\\\', '', text)
    text = re.sub(r'\\n', '\n', text)
    
    return text

def escape_html(text):
    """Escape HTML special characters"""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

def trim_display_text(prefix, highlight, suffix):
    """Trim leading/trailing whitespace while preserving paragraph breaks"""
    # Remove leading whitespace from prefix (except preserve one newline if multiple exist)
    prefix_stripped = prefix.lstrip(' \t')
    if prefix.startswith('\n'):
        # Count leading newlines
        leading_newlines = len(prefix) - len(prefix.lstrip('\n'))
        if leading_newlines > 1:
            prefix_stripped = '\n' + prefix_stripped
    
    # Remove trailing whitespace from suffix (except preserve one newline if multiple exist)
    suffix_stripped = suffix.rstrip(' \t')
    if suffix.endswith('\n'):
        trailing_newlines = len(suffix) - len(suffix.rstrip('\n'))
        if trailing_newlines > 1:
            suffix_stripped = suffix_stripped + '\n'
    
    return prefix_stripped, highlight, suffix_stripped

def detect_content_type(text):
    """
    Detect if text contains tables or equations.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with 'in_table' and 'in_equation' boolean flags
    """
    in_table = '|' in text
    in_equation = bool(re.search(r'\$|\\\[|\\\(|\\mathrm|\\mathsf|\\text|\\frac', text))
    
    return {
        'in_table': in_table,
        'in_equation': in_equation
    }
    
def extract_sentence_matches(
    context, 
    target_sentence, 
    window_size_padding=5, 
    score_thr=50,
    context_words=10,
    max_results=3
):
    """
    Extract matching text segments from context based on fuzzy matching.
    
    Args:
        context: Source markdown text to search in
        target_sentence: Text to search for
        window_size_padding: Extra words to include in search window
        score_thr: Minimum fuzzy match score (0-100)
        context_words: Number of words to show before/after match for context
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        List of dictionaries, each containing:
            - score: Match confidence (0-100)
            - prefix: Text before the match
            - highlight: The matched text
            - suffix: Text after the match
            - in_table: Boolean indicating if match is in a table
            - in_equation: Boolean indicating if match contains equations
            - start_idx: Token start index
            - end_idx: Token end index
            
    Example:
        >>> results = extract_sentence_matches(text, "nitrogen emissions")
        >>> for r in results:
        ...     print(f"Score: {r['score']}%, Match: {r['highlight']}")
    """
    
    # Tokenize while preserving whitespace
    tokens = re.split(r'(\s+)', context)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    
    target_clean = clean_for_matching(target_sentence)
    target_word_count = len([w for w in target_sentence.split() if w.strip()])
    
    # Adaptive window size based on target length
    window_size_words = max(target_word_count + window_size_padding, 10)
    
    results = []
    
    # Sliding window search
    for i in range(len(word_indices) - window_size_words + 1):
        start_idx = word_indices[i]
        end_idx = word_indices[min(len(word_indices) - 1, i + window_size_words - 1)]
        
        # Extract window text
        window_text_raw = "".join(tokens[start_idx : end_idx + 1])
        window_clean = clean_for_matching(window_text_raw)
        
        # Fuzzy match score
        score = fuzz.ratio(target_clean, window_clean)
        
        if score > score_thr:
            # Expand context for display
            disp_start_idx = max(0, i - context_words)
            disp_end_idx = min(len(word_indices) - 1, i + window_size_words + context_words)
            
            disp_start = word_indices[disp_start_idx]
            disp_end = word_indices[disp_end_idx]
            
            # Extract text segments
            prefix = "".join(tokens[disp_start : start_idx])
            highlight = window_text_raw
            suffix = "".join(tokens[end_idx + 1 : disp_end + 1])
            
            # Trim whitespace
            prefix, highlight, suffix = trim_display_text(prefix, highlight, suffix)
            
            # Detect content type
            content_type = detect_content_type(prefix + highlight + suffix)
            
            results.append({
                "score": round(score, 2),
                "prefix": prefix,
                "highlight": highlight,
                "suffix": suffix,
                "in_table": content_type['in_table'],
                "in_equation": content_type['in_equation'],
                "start_idx": start_idx,
                "end_idx": end_idx
            })
    
    # Sort by score (highest first)
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Remove overlapping results
    filtered_results = []
    used_ranges = []
    for res in results:
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (res['end_idx'] < used_start or res['start_idx'] > used_end):
                overlaps = True
                break
        if not overlaps:
            filtered_results.append(res)
            used_ranges.append((res['start_idx'], res['end_idx']))
            if len(filtered_results) >= max_results:
                break
    
    return filtered_results

def sentence_match_display(
    context=None, 
    target_sentence=None, 
    window_size_padding=5, 
    score_thr=50,
    context_words=10,
    max_results=3,
    filtered_results=None
):
    """
    Search and display highlighted text matches in markdown content.
    
    Args:
        context: Source markdown text
        target_sentence: Text to search for
        window_size_padding: Extra words to include in search window
        score_thr: Minimum fuzzy match score (0-100)
        context_words: Number of words to show before/after match
        max_results: Maximum number of results to display
    """
    from IPython.display import display, HTML  # notebook-only; keep out of the module's import path

    if not filtered_results:
        # Extract matches using the extraction function
        try:
            filtered_results = extract_sentence_matches(
                context=context,
                target_sentence=target_sentence,
                window_size_padding=window_size_padding,
                score_thr=score_thr,
                context_words=context_words,
                max_results=max_results
            )
        except ValueError as e:
            print(f"Logging error: {e}") # Log the error
            raise(e)
    
    # Display results
    if not filtered_results:
        print(f"No matches found above {score_thr}% threshold")
        return
    
    # Display target
    display(HTML(f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin-bottom: 25px; padding: 15px; 
                background: #f0f7ff; 
                border-radius: 6px;
                border-left: 5px solid #0066cc;">
        <div style="font-size: 0.9em; color: #0066cc; font-weight: 600; margin-bottom: 8px;">
            🔍 Searching for:
        </div>
        <div style="font-size: 1em; color: #333; line-height: 1.6; font-style: italic;">
            "{escape_html(target_sentence)}"
        </div>
        <div style="font-size: 0.8em; color: #666; margin-top: 8px;">
            Found {len(filtered_results)} match{'es' if len(filtered_results) != 1 else ''} above {score_thr}% threshold
        </div>
    </div>
    """))
    
    # Display matches
    for idx, res in enumerate(filtered_results):
        if idx >= max_results:
            continue
        # Adaptive styling
        if res['in_equation']:
            highlight_style = "background-color: #e7f3ff; border-bottom: 2px solid #2196F3; padding: 2px 4px; font-weight: 500;"
            border_color = "#2196F3"
        elif res['in_table']:
            highlight_style = "background-color: #f0f8ff; border-bottom: 2px solid #4CAF50; padding: 2px 4px; font-weight: 500;"
            border_color = "#4CAF50"
        else:
            highlight_style = "background-color: #fff3cd; border-bottom: 2px solid #ffc107; padding: 2px 4px; font-weight: 500;"
            border_color = "#ffc107"
        
        # Render and escape
        prefix_rendered = escape_html(render_latex_for_display(res['prefix']))
        highlight_rendered = escape_html(render_latex_for_display(res['highlight']))
        suffix_rendered = escape_html(render_latex_for_display(res['suffix']))
        
        # Build HTML
        html_output = (
            prefix_rendered +
            f"<span style='{highlight_style}'>{highlight_rendered}</span>" +
            suffix_rendered
        )
        
        html_output = html_output.replace('\n\n', '<br><br>').replace('\n', '<br>')
        
        match_label = "Match" if len(filtered_results) == 1 else f"Match {idx + 1}"
        
        display(HTML(f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    line-height: 1.8; color: #333; margin-bottom: 20px;">
            <div style="font-size: 0.85em; color: #666; margin-bottom: 8px; font-weight: 500;">
                {match_label} — Confidence: {res['score']}%
                {' 📊 (in table)' if res['in_table'] else ''}
                {' 📐 (with equations)' if res['in_equation'] else ''}
            </div>
            <div style="background: #f8f9fa; 
                        border-left: 5px solid {border_color}; 
                        padding: 18px; 
                        border-radius: 4px; 
                        overflow-x: auto;
                        font-size: 0.95em;">
                {html_output}
            </div>
        </div>
        """))