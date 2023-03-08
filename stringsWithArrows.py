def stringsWithArrows(text, posStart, posEnd):
    result = ''

    # Calculate indices
    idxStart = max(text.rfind('\n', 0, posStart.idx), 0)
    idxEnd = text.find('\n', idxStart + 1)
    if idxEnd < 0: idxEnd = len(text)

    # Generate each line
    lineCount = posEnd.ln - posStart.ln + 1
    for i in range(lineCount):
        # Calculate line columns
        line = text[idxStart:idxEnd]
        colStart = posStart.col if i == 0 else 0
        colEnd = posEnd.col if i == lineCount - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * colStart + '^' * (colEnd - colStart)

        # Re-calculate indices
        idxStart = idxEnd
        idxEnd = text.find('\n', idxStart + 1)
        if idxEnd < 0: idxEnd = len(text)

    return result.replace('\t', '')