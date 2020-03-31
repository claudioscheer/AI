import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


def find_files(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in find_files("data/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError(
        "Data not found. Make sure that you downloaded data "
        "from https://download.pytorch.org/tutorial/data.zip and extract it to "
        "the current directory."
    )

print("# categories:", n_categories, all_categories)
print(unicode_to_ascii("O'Néàl"))
