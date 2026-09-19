import os
import re

def convert_conllu_comments_to_txt(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".conllu"):
            conllu_path = os.path.join(input_dir, filename)
            txt_path = os.path.join(output_dir, filename.replace(".conllu", ".txt"))

            with open(conllu_path, 'r', encoding='utf-8') as infile, \
                 open(txt_path, 'w', encoding='utf-8') as outfile:

                for line in infile:
                    line = line.strip()
                    if line.startswith("#") and "=" in line:
                        # Match: # 7. Hawaiian sentence = English translation
                        match = re.match(r"#\s*\d+\.\s*(.+?)\s*=\s*.*", line)
                        if match:
                            hawaiian_sentence = match.group(1).strip()
                            outfile.write(hawaiian_sentence + "\n")

convert_conllu_comments_to_txt("conllu_files", "text_files")
