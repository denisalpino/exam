import os
from typing import Dict
from ast import literal_eval

import camelot
import requests
import pandas as pd
import trafilatura
from markdownify import markdownify as md

from common.settings import settings


def html2md(html: str) -> str:
    # Extract only representative parts
    good_html = trafilatura.extract(
        html,
        include_formatting=True, include_tables=True,
        favor_recall=True, include_links=True
    )

    # Return empty string whether there is not enough represntative tag
    # with text in the current HTML
    if not good_html:
        return ""

    # Convert HTML to rh Markdown format
    markdown = md(good_html, heading_style="ATX")

    # Return empty string whether there is not enough charecters
    # because in such cases we often get CAPTCHAs
    if len(markdown) < 200:
        return ""
    return markdown


def to_sem_list(s: str):
    if pd.isna(s):
        return []
    parts = [p for p in ''.join(ch if ch.isdigit() or ch==',' else ' ' for ch in str(s)).split() if p]
    # split on commas too
    flat = []
    for p in parts:
        flat += [q for q in p.split(',') if q]
    try:
        return sorted({int(x) for x in flat})
    except Exception:
        return []


def pdf2df(path: str) -> pd.DataFrame:
    # Read and parse pdf
    dfs = [table.df for table in camelot.read_pdf(path, pages="1-end")] # type: ignore

    # Concatenate tables from pages
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Rename cols
    new_df = df.rename(columns={
        df.columns[0]: "semester",   # was '0'
        df.columns[1]: "title",      # was '1' (original text/title as-is)
        df.columns[2]: "credits",    # was '2'
        df.columns[3]: "hours"       # was '3'
    })

    new_df["semester_list"] = new_df["semester"].apply(to_sem_list)
    new_df.drop(columns=["semester"])

    return new_df


def df2structured_dicts(df: pd.DataFrame) -> Dict:
    # Convert df to dict of dfs (by blocks)
    blocks = {}
    current_block_name = None
    current_rows = []

    for _, row in df.iterrows():
        if isinstance(row["title"], str) and row["title"].strip().startswith("Блок"):
            # Если уже был блок — сохраняем его перед началом нового
            if current_block_name is not None:
                blocks[current_block_name] = pd.DataFrame(current_rows, columns=df.columns)
            # starts the new block
            current_block_name = row["title"].strip()
            current_rows = []
        else:
            if current_block_name is not None:
                current_rows.append(row.tolist())

    # add last block
    if current_block_name is not None and current_rows:
        blocks[current_block_name] = pd.DataFrame(current_rows, columns=df.columns)

    # Convert df to dict dicts of dfs (by blocks and subblocks)
    nested_blocks = {}

    for block_name, block_df in blocks.items():
        sub_dict = {}
        current_title = None
        current_rows = []

        # Приводим semester_list к list, если он хранится как строка
        block_df["semester_list"] = block_df["semester_list"].apply(
            lambda x: literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )

        for _, row in block_df.iterrows():
            if isinstance(row["semester_list"], list) and len(row["semester_list"]) == 0:
                # Сохраняем предыдущий подраздел, если был
                if current_title is not None:
                    sub_dict[current_title] = pd.DataFrame(current_rows, columns=block_df.columns)
                # Новый подзаголовок
                current_title = row["title"].strip()
                current_rows = []
            else:
                if current_title is not None:
                    current_rows.append(row.tolist())

        # Добавляем последний подраздел
        if current_title is not None and current_rows:
            sub_dict[current_title] = pd.DataFrame(current_rows, columns=block_df.columns)

        nested_blocks[block_name] = sub_dict

    return nested_blocks


def dct2txt(dct: Dict) -> str:
    # Convert to text
    program = []
    for bname, sub in dct.items():
        program.append(f"## {bname}")
        for subname, disciplines in sub.items():
            program.append(f"### {subname}")
            for ind, row in disciplines.iterrows():
                program.append(
                    f"{ind + 1}. {row['title']}\n"
                    f"- Количество кредитов, начисляемых за дисциплину: {row['credits']}\n"
                    f"- Сумма учебных часов: {row['hours']}\n"
                    F"- Номера семестров: {', '.join(map(str, row['semester_list']))}\n"
                )
    program = "\n".join(program)

    return program


def parse():
    base_dir = settings.DATA_DIR
    os.makedirs(base_dir, exist_ok=True)

    pages = {
        "ai": "https://abit.itmo.ru/program/master/ai",
        "ai_product": "https://abit.itmo.ru/program/master/ai_product"
    }

    for fname, url in pages.items():
        # Скачиваем HTML
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers).text

        # Конвертируем в Markdown
        markdown_content = html2md(html)

        # Сохраняем содержимое сайта
        with open(os.path.join(base_dir, f"{fname}.txt"), "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Обрабатываем PDF
        pdf_path = os.path.join(base_dir, f"{fname}.pdf")
        if os.path.exists(pdf_path):
            df = pdf2df(pdf_path)
            dct = df2structured_dicts(df)
            txt_content = dct2txt(dct)
            with open(os.path.join(base_dir, f"{fname}_program.txt"), "w", encoding="utf-8") as f:
                f.write(txt_content)
        else:
            print(f"PDF not found: {pdf_path}")
