import copy
import json
import os
import re
import string
import traceback
import uuid
from collections import Counter, defaultdict
from io import StringIO
from typing import Union

synonyms = {
    'Hydrogen': 'H',
    'Helium': 'He',
    'Lithium': 'Li',
    'Beryllium': 'Be',
    'Boron': 'B',
    'Carbon': 'C',
    'Nitrogen': 'N',
    'Oxygen': 'O',
    'Fluorine': 'F',
    'Neon': 'Ne',
    'Sodium': 'Na',
    'Magnesium': 'Mg',
    'Aluminium': 'Al',
    'Aluminium(aluminum)': 'Al',
    'Silicon': 'Si',
    'Phosphorus': 'P',
    'Sulfur': 'S',
    'Chlorine': 'Cl',
    'Argon': 'Ar',
    'Potassium': 'K',
    'Calcium': 'Ca',
    'Scandium': 'Sc',
    'Titanium': 'Ti',
    'Vanadium': 'V',
    'Chromium': 'Cr',
    'Manganese': 'Mn',
    'Iron': 'Fe',
    'Cobalt': 'Co',
    'Nickel': 'Ni',
    'Copper': 'Cu',
    'Zinc': 'Zn',
    'Gallium': 'Ga',
    'Germanium': 'Ge',
    'Arsenic': 'As',
    'Selenium': 'Se',
    'Bromine': 'Br',
    'Krypton': 'Kr',
    'Rubidium': 'Rb',
    'Strontium': 'Sr',
    'Yttrium': 'Y',
    'Zirconium': 'Zr',
    'Niobium': 'Nb',
    'Molybdenum': 'Mo',
    'Technetium': 'Tc',
    'Ruthenium': 'Ru',
    'Rhodium': 'Rh',
    'Palladium': 'Pd',
    'Silver': 'Ag',
    'Cadmium': 'Cd',
    'Indium': 'In',
    'Tin': 'Sn',
    'Antimony': 'Sb',
    'Tellurium': 'Te',
    'Iodine': 'I',
    'Xenon': 'Xe',
    'Cesium': 'Cs',
    'Barium': 'Ba',
    'Lanthanum': 'La',
    'Cerium': 'Ce',
    'Praseodymium': 'Pr',
    'Neodymium': 'Nd',
    'Promethium': 'Pm',
    'Samarium': 'Sm',
    'Europium': 'Eu',
    'Gadolinium': 'Gd',
    'Terbium': 'Tb',
    'Dysprosium': 'Dy',
    'Holmium': 'Ho',
    'Erbium': 'Er',
    'Thulium': 'Tm',
    'Ytterbium': 'Yb',
    'Lutetium': 'Lu',
    'Hafnium': 'Hf',
    'Tantalum': 'Ta',
    'Tungsten': 'W',
    'Rhenium': 'Re',
    'Osmium': 'Os',
    'Iridium': 'Ir',
    'Platinum': 'Pt',
    'Gold': 'Au',
    'Mercury': 'Hg',
    'Thallium': 'Tl',
    'Lead': 'Pb',
    'Bismuth': 'Bi',
    'Polonium': 'Po',
    'Astatine': 'At',
    'Radon': 'Rn',
    'Francium': 'Fr',
    'Radium': 'Ra',
    'Actinium': 'Ac',
    'Thorium': 'Th',
    'Protactinium': 'Pa',
    'Uranium': 'U',
    'Neptunium': 'Np',
    'Plutonium': 'Pu',
    'Americium': 'Am',
    'Curium': 'Cm',
    'Berkelium': 'Bk',
    'Californium': 'Cf',
    'Einsteinium': 'Es',
    'Fermium': 'Fm'
}


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_normalize_name(s):
    if s.startswith("Unnamed"):
        return ""
    else:
        """ Standardize name or index string. """
        # # 定义需要移除的单位和符号
        # units = ["µM", "µg/mL", "nM", "%", "wt.%", "at.%", "at%", "wt%"]
        # for unit in units:
        #     s = s.replace(unit, "")

        # 定义特定关键字
        keywords = ["pIC50", "IC50", "EC50", "TC50", "GI50", "Ki", "Kd", "Kb", "pKb"]

        # 移除非字母数字的字符，除了空格
        s = re.sub(r'[^\w\s%.\-\(\)]', '', s)
        if s in synonyms:
            s = synonyms[s]

        # 分割字符串为单词列表
        words = s.split()

        # 将关键字移到末尾
        reordered_words = [word for word in words if word not in keywords]
        keywords_in_string = [word for word in words if word in keywords]
        reordered_words.extend(keywords_in_string)
        # 重新组合为字符串
        return ' '.join(reordered_words)


def fuzzy_normalize_value(vi):
    try:
        vi = str(vi).lower()

        if "bal" in vi or "remainder" in vi or "bas" in vi:
            vi = "bal"
            return "bal"

        if ("nan" in vi and not "–" in vi) or "/" == vi or "n/a" in vi or "na" in vi or vi == "":
            vi = "0"
        vi = vi.replace("nan", "–").replace("~", "-")

        pattern = r"\d+(?:\.\d+)?"
        matches = re.findall(pattern, vi)
        if len(matches) == 2:
            vi = f"{matches[0]}-{matches[1]}"
        elif len(matches) == 1:
            vi = matches[0]

        if "<" in vi:
            vi = vi.replace("<", "")
        if ">" in vi:
            vi = vi.replace(">", "")

        try:
            vi = float(vi)
            vi = round(vi, 3)
        except:
            # print(vi)
            pass
    except:
        pass

    return vi


code_pattern = r"```[\s\S]*?\n([\s\S]+?)\n```"
json_pattern = r"```json[\s\S]*?\n([\s\S]+?)\n```"
csv_pattern = r"```csv[\s\S]*?\n([\s\S]+?)\n```"
table_pattern = r"^({index0}[\s\S]+)[`]*"
outlink_pattern = r"\[Download[a-zA-Z0-9 ]+?\]\((https://[a-zA-Z0-9_. /]+?)\)"


def extract_table(sampled: str, format: str = "csv",
                  index: Union[str, list, tuple] = ("Compound", ""), compare_fields: list = [],
                  **kwargs):
    import pandas as pd

    def parse_csv_text(csvtext: str) -> str:
        lines = csvtext.strip().split("\n")
        tuple_pattern = r"\((\"[\s\S]*?\"),(\"[\s\S]*?\")\)"
        if re.search(tuple_pattern, lines[0]) is not None:
            lines[0] = re.sub(tuple_pattern, r"(\1|\2)", lines[0])
        lines_clr = [re.sub(r"\"[\s\S]*?\"", "", line) for line in lines]
        max_commas = max([line_clr.count(",") for line_clr in lines_clr])
        unified_lines = [line + ("," * (max_commas - line_clr.count(","))) for line, line_clr in zip(lines, lines_clr)]
        return "\n".join(unified_lines)

    def parse_table_multiindex(table: pd.DataFrame, compare_fields: list = []) -> pd.DataFrame:
        """
        Parse a table with multiindex columns.
        """

        df = table.copy()
        if df.columns.nlevels == 1 and tuple in [type(f) for f in compare_fields]:
            coltypes = {col: type(df[col].iloc[0]) for col in df.columns}
            for col, ctype in coltypes.items():
                if ctype == str:
                    if ":" in df[col].iloc[0] and "," in df[col].iloc[0]:
                        df[col] = [{key: value for key, value in [pair.split(": ") for pair in data.split(", ")]} for
                                   data
                                   in df[col]]
                        coltypes[col] = dict
            dfs = []

            for col, ctype in coltypes.items():
                if ctype == dict:
                    d = pd.DataFrame(df.pop(col).tolist())
                    d.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize_name(key)) for key in d.columns])
                    dfs.append(d)
            df.columns = pd.MultiIndex.from_tuples(
                [eval(col.replace("|", ",")) if (col[0] == "(" and col[-1] == ")") else
                 (col, "") for col in df.columns])
            df = pd.concat([df] + dfs, axis=1)
        if df.columns.nlevels > 1:
            df.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize_name(subcol)) for col, subcol in df.columns])

        return df

    compare_fields_types = [type(x) for x in compare_fields]
    header_rows = [0, 1] if tuple in compare_fields_types else [0]

    try:
        if re.search(outlink_pattern, sampled) is not None:
            code = re.search(outlink_pattern, sampled).group()
            link = re.sub(outlink_pattern, r"\1", code)

            fname = f"/tmp/LLMEvals_{uuid.uuid4()}.csv"
            os.system(f"wget {link} -O {fname}")
            table = pd.read_csv(fname)
            if pd.isna(table.iloc[0, 0]):
                table = pd.read_csv(fname, header=header_rows)
        elif format == "csv":
            starts = index if type(index) == str else index[0]
            table_pattern_format = table_pattern.format(index0=starts)
            if re.search(csv_pattern, sampled) is not None:
                code = re.search(csv_pattern, sampled).group()
                code_content = re.sub(csv_pattern, r"\1", code)

            elif re.search(table_pattern_format, "\n" + sampled) is not None:
                code = re.search(table_pattern_format, "\n" + sampled).group().strip()
                code_content = re.sub(table_pattern_format, r"\1", code)
            else:
                code_content = sampled
            code_content_processed = parse_csv_text(code_content)
            # table = pd.read_csv(StringIO(code_content_processed), header=header_rows)
            table = pd.read_csv(StringIO(code_content_processed))
            if table.shape[0] == 0:
                table = pd.DataFrame()
            elif pd.isna(table.iloc[0, 0]):
                table = pd.read_csv(StringIO(code_content_processed), header=header_rows)

        elif format == "json":
            code = re.search(json_pattern, sampled).group()
            code_content = re.sub(json_pattern, r"\1", code).replace("\"", "")
            table = pd.DataFrame(json.loads(code_content))
        else:
            table = pd.DataFrame()
        table = parse_table_multiindex(table, compare_fields=compare_fields)

        if table.shape[0] != 0:
            idxlist = table.columns
            if type(index) == str and table.columns.nlevels > 1:
                index = tuple([index] + ["" for _ in range(table.columns.nlevels - 1)])
            if type(index) in [str, tuple]:
                if index not in table.columns:
                    idxlist = [index] + list(table.columns)[1:]
            elif type(index) == list:
                if True in [idx not in table.columns for idx in index]:
                    idxlist = list(index) + list(table.columns)[len(index):]
            table.columns = idxlist if table.columns.nlevels == 1 else pd.MultiIndex.from_tuples(idxlist)
        # answerfile_out = kwargs["answerfile_name"].replace(".csv", "_output.csv")
        # table.to_csv(answerfile_out, index=False)
        # picked_str = open(answerfile_out, 'r').read()
    except:
        traceback.print_exc()
        table = None
        # picked_str = "Failed to parse"
    return table


def extract_or_validate_choice(sampled, **kwargs):
    # First rule: Extract a choice and its associated value
    pattern = re.compile(r'\w\)\s\d+(?:\.\d+)?(?:\s?:\s?\d+(?:\.\d+)?)?\s?[°]?[CK]?')
    matches = pattern.findall(sampled)
    if matches:
        # Process the first match to normalize temperature representations
        sampled0 = matches[0].replace("°", " ").replace("  ", " ")
        return sampled0

    # If the first rule didn't find a match, proceed with the second rule
    # Second rule: Validate choice based on multiline structure and `ideal` criteria
    lines = sampled.split("\n")
    for line in reversed(lines):
        if line.strip() == "":
            continue
        for option in ["a)", "b)", "c)", "d)"]:
            if option in kwargs.get("ideal", []) and option in line:
                # If the line matches an 'ideal' option, return it
                return line
            elif option not in kwargs.get("ideal", []) and option not in line:
                continue
            else:
                break
    # If neither rule yields a result, return "No answer."
    return sampled


def extract_yes_or_no(sampled, **kwargs):
    if re.search(r"^Yes", sampled, re.IGNORECASE):
        answer = "Yes"
    elif re.search(r"^No", sampled, re.IGNORECASE):
        answer = "No"
    elif re.search(r"^Maybe", sampled, re.IGNORECASE):
        answer = "Maybe"
    elif "Yes" in sampled or "yes" in sampled:
        answer = "Yes"
    elif "No" in sampled or "no" in sampled:
        answer = "No"
    elif "Maybe" in sampled or "maybe" in sampled:
        answer = "Maybe"
    else:
        answer = sampled
    return answer


def extract_entities(input_str, **kwargs):
    """
    Extracts entities from the input string that are formatted as (...), (...), (...).

    Args:
    - input_str (str): The input string containing entities.

    Returns:
    - list of entities: A list containing all extracted entities.
    """
    pattern = r'\(\s*([^,]+)\s*\)'
    try:
        matches = re.findall(pattern, input_str)
        return matches
    except:
        print(f"cannot extract entities from {input_str}")
        return None


def extract_tuples(input_str, **kwargs):
    """
    Extracts tuples from the input string that are formatted as (element1, element2),
    allowing for variable spacing between elements.

    Args:
    - input_str (str): The input string containing pairs.

    Returns:
    - list of tuples: A list containing all extracted pairs.
    """
    pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)\s*\)'
    try:
        matches = re.findall(pattern, input_str)
        return matches
    except:
        print(f"cannot extract tuples from {input_str}")
        return []


def extract_triplets(input_str, **kwargs):
    """
    Extracts tuples from the input string that are formatted as (..., ..., ...).

    Args:
    - input_str (str): The input string containing tuples.

    Returns:
    - list of tuples: A list containing all extracted tuples.
    """
    # 正则表达式用于匹配格式为 (..., ..., ...) 的三元组
    pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)'
    try:
        matches = re.findall(pattern, input_str)
        return matches
    except:
        print(f"cannot extract triplets from {input_str}")
        return []
