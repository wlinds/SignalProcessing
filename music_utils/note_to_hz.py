def create_md(base_frequencies, file_name='notes.md', max_col=10):
    if len(base_frequencies) != 12:
        raise ValueError("Input list must contain exactly 12 base frequencies.")
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    header = "| | " + " | ".join(str(i) for i in range(1, max_col + 1)) + " |\n"
    separator = "| --- " + " | " + " - | " * max_col + " \n"
    
    body = ""
    for i, note in enumerate(notes):
        row = f"| {note} "
        frequency = base_frequencies[i]
        for _ in range(max_col):
            row += f"| {frequency:.2f} "
            frequency *= 2
        row += "|\n"
        body += row
    
    c = header + separator + body
    
    with open(file_name, 'w') as file:
        file.write(c)

base_frequencies = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87]
create_md(base_frequencies)
