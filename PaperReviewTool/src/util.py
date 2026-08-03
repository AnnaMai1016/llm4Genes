import re
import tiktoken

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def count_tokens(text, model_name="gpt-4o-mini"):
    """Counts the number of tokens in a text string for a given model."""
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def read_full_paper(filename):
    # This function is used to read Mineru Generated Markdown file.
    # The sections not relevant to paper content are removed.
    # stop_list = set([
    #     'acknowledgements', 'references', 'supporting information', 
    #     'statement', 'funding', 'acknowledgments'])
    stop_list = set([
        'acknowledgements', 'references', 'supporting information', 
        'statement', 'acknowledgments'])
    with open(filename, 'r') as f:
        full_text = f.readlines()
    flag = False
    for idx, line in enumerate(full_text):
        # Match any markdown heading level ('# ', '## ', ...), not just H1 —
        # this MinerU version emits section headers like '## REFERENCES' (H2);
        # only the paper title itself is H1.
        stripped = line.lstrip('#')
        if len(stripped) < len(line) and stripped[:1] == ' ':
            for item in stop_list:
                if item in line.lower():
                    flag = True
        if flag:
            break
    full_text = ''.join(full_text[:idx])
    
    figure_pattern = re.compile(r'!\[]\(.*?.jpg\)')
    full_text = figure_pattern.sub('', full_text)
    
    return full_text


# sentence_match_display is imported directly from sentence_match above.
