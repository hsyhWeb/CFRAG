import json
import os

# 你下载的原始数据文件路径
raw_data = {
    "train": "data/LaMP-QA/train.json",
    "dev": "data/LaMP-QA/validation.json",
    "test": "data/LaMP-QA/test.json"
}

# 输出文件夹路径
output_root = "data/LaMP-QA"

def convert_data(phase):
    with open(raw_data[phase], "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    outputs = {"task": "LaMP-QA", "golds": []}

    for item in data:
        q_id = item["id"]
        questions.append({
            "id": q_id,
            "input": item.get("question", item.get("text", "")),
            "profile": item.get("profile", []),
            "rubric_aspects": item.get("rubric_aspects", []),
            "narrative": item.get("narrative", ""),
            "category": item.get("category", "")
        })

        outputs["golds"].append({
            "id": q_id,
            "output": item.get("narrative", "")
        })

    # 创建输出目录
    out_dir = os.path.join(output_root, phase)
    os.makedirs(out_dir, exist_ok=True)

    # 写入文件
    questions_file = os.path.join(out_dir, f"{phase}_questions.json")
    outputs_file = os.path.join(out_dir, f"{phase}_outputs.json")

    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4, ensure_ascii=False)

    with open(outputs_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    print(f"{phase} -> {questions_file}, {outputs_file}")

if __name__ == "__main__":
    for phase in ["train", "dev", "test"]:
        convert_data(phase)
