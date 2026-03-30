with open('rf_alpha_comparison.md', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

table_lines = [l for l in lines if '|' in l]
with open('output_depth.txt', 'w', encoding='utf-8') as f:
    f.writelines(table_lines)
