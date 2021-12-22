# distant.py
# Author: Kristian N Jensen
# Date: 08/11-2021
# Methods for Distant supervision

import os
import csv
import sys
import requests
import json
import math
from loguru import logger
from Levenshtein import distance as lev
from typing import Dict, NamedTuple, Tuple
from tqdm import tqdm


class Skill(NamedTuple):
    annotated_skill: str
    annotated_type: str
    esco_skill: str
    esco_type: str
    esco_tax: str

    def __str__(self):
        return (
            f"Annotated Skill: {self.annotated_skill}\n"
            f"Annotated Type: {self.annotated_type}\n"
            f"ESCO Skill: {self.esco_skill}\n"
            f"ESCO Type: {self.esco_type}\n"
            f"ESCO Tax: {self.esco_tax}\n"
        )


def skill_to_csv(skill: Skill) -> str:
    return f"{skill.annotated_skill},{skill.annotated_type},{skill.esco_skill},{skill.esco_type},{skill.esco_tax}"


def get_skill_tax(entry: Dict) -> str:
    code = entry.get("broaderHierarchyConcept")[0].split("/")[-1].split(".")[0]
    return code


def get_knowledge_tax(entry: Dict) -> str:
    k_tag = len(entry.get("broaderHierarchyConcept")[0].split("/")[-1])
    if k_tag == 4:
        code = f'K{entry.get("broaderHierarchyConcept")[0][-4:-2]}'
    elif k_tag == 3:
        code = f'K{entry.get("broaderHierarchyConcept")[0][-3:-1]}'
    else:
        code = entry.get("broaderHierarchyConcept")[0].split("/")[-1]
    return code


def entry_to_skill(annotated: Tuple, entry: Dict) -> Skill:
    found_type = entry.get("hasSkillType")[0].split("/")[-1]
    try:
        tax = (
            get_skill_tax(entry) if found_type == "skill" else get_knowledge_tax(entry)
        )
    except (IndexError, TypeError):
        tax = "K99"
    skill = Skill(annotated[0], annotated[1], entry["title"].lower(), found_type, tax)
    return skill


def get_skill(skill: str, type_: str, lang: str = "en") -> Skill:
    params = dict(
        text=str(skill), type="skill", language=lang, limit=100, offset=0, full=False
    )

    request = requests.get(url=f"http://localhost:8080/search?", params=params)
    response = json.loads(request.text)
    results = response["_embedded"]["results"]

    if len(results) == 0:
        logger.warning(f"No results for {skill} in ESCO")
        return Skill(skill, type_, "", "", "")

    results = list(
        map(lambda x: entry_to_skill((skill.lower(), type_.lower()), x), results)
    )
    results_ = list(
        filter(
            lambda x: x.annotated_type == x.esco_type and x.esco_tax != "0000",
            results,
        )
    )
    if len(results_) < 1:
        logger.warning(
            f"ESCO does not have a skill/knowledge with the same type as {skill}"
        )
    else:
        results = results_

    min_dist = (None, math.inf)
    for _, result in enumerate(results):
        dist = lev(result.annotated_skill, result.esco_skill)
        if result.annotated_skill in result.esco_skill:
            return result
        elif dist < min_dist[1]:
            min_dist = (result, dist)

    return min_dist[0]


def get_skills(path: str, lang: str):
    csv_path = "".join(path.split(".")[:-1]) + ".esco.csv"
    txt_path = "".join(path.split(".")[:-1]) + ".esco.jsonl"
    with open(path, "r") as skill_file, open(csv_path, "w") as csv_file, open(
        txt_path, "w"
    ) as txt_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(
            [
                "Annotated Skill",
                "Annotated Type",
                "ESCO Skill",
                "ESCO Type",
                "ESCO Taxonomy",
            ]
        )
        for line in tqdm(skill_file.readlines()):
            skill, type_ = line.strip().split("\t")
            found_skill = get_skill(skill.rstrip(".,;:)({} "), type_, lang)
            csv_writer.writerow(list(found_skill))  # skill_to_csv(found_skill) + "\n")
            jsonl = {
                "text": str(found_skill),
                "labels": [
                    found_skill.esco_tax if found_skill.esco_tax != "" else "K99"
                ],
            }
            txt_file.write(json.dumps(jsonl) + "\n")


if __name__ == "__main__":
    # Path to skills file "Skill\tType"
    path = sys.argv[1]
    # Language of the skills, e.g. en
    lang = sys.argv[2]
    get_skills(path, lang)