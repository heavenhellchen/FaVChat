from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from rouge_score import rouge_scorer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.trainer import FaVChatGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified


@dataclass
class FaVChatGRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["facial_reward", "format"],
        metadata={"help": "Reward functions used by FaVChat GRPO."},
    )
    max_pixels: Optional[int] = field(default=12845056, metadata={"help": "Maximum number of pixels."})
    min_pixels: Optional[int] = field(default=3136, metadata={"help": "Minimum number of pixels."})
    temporal: Optional[bool] = field(default=True, metadata={"help": "Enable temporal reward shaping."})
    len_control: Optional[bool] = field(default=True, metadata={"help": "Enable length control reward."})
    use_de_grpo: Optional[bool] = field(default=True, metadata={"help": "Enable DE-GRPO."})
    data_root: Optional[str] = field(default="./FaVChat-data", metadata={"help": "Dataset root path."})
    remove_threshold: Optional[float] = field(default=0.20, metadata={"help": "DE-GRPO remove threshold."})
    keep_threshold: Optional[float] = field(default=0.80, metadata={"help": "DE-GRPO keep threshold."})
    utility_smoothing: Optional[float] = field(default=0.80, metadata={"help": "DE-GRPO smoothing lambda."})
    reward_decay: Optional[float] = field(default=0.50, metadata={"help": "DE-GRPO decay delta."})
    attr_weight: Optional[float] = field(default=0.40, metadata={"help": "Attribute reward weight."})
    emo_weight: Optional[float] = field(default=0.30, metadata={"help": "Emotion reward weight."})
    act_weight: Optional[float] = field(default=0.30, metadata={"help": "Action reward weight."})


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please reason carefully about subtle facial appearance, expression, and motion cues. "
    "Write your reasoning inside <think> </think> and the final response inside <answer> </answer>."
)

TYPE_TEMPLATE = {
    "multiple choice": " Return only the single option letter inside <answer> </answer>.",
    "numerical": " Return only the numerical value inside <answer> </answer>.",
    "OCR": " Transcribe the relevant facial/video text inside <answer> </answer>.",
    "free-form": " Return a concise natural-language answer inside <answer> </answer>.",
    "regression": " Return only the numerical value inside <answer> </answer>.",
}


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def tokenize_for_similarity(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize_for_similarity(left))
    right_tokens = set(tokenize_for_similarity(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def get_facial_target(kwargs, key: str, index: int) -> str:
    values = kwargs.get(key)
    if not values:
        return ""
    if index < len(values):
        return values[index] or ""
    return values[0] or ""


def facial_reward(completions, solution=None, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    attr_weight = kwargs.get("attr_weight", [0.4])[0]
    emo_weight = kwargs.get("emo_weight", [0.3])[0]
    act_weight = kwargs.get("act_weight", [0.3])[0]
    rewards = []
    for index, content in enumerate(contents):
        answer = extract_answer(content)
        attr_target = get_facial_target(kwargs, "facial_attributes", index)
        emo_target = get_facial_target(kwargs, "facial_emotion", index)
        act_target = get_facial_target(kwargs, "facial_action", index)
        if not any([attr_target, emo_target, act_target]):
            gt = extract_answer(solution[index]) if solution is not None else ""
            rewards.append(jaccard_similarity(answer, gt))
            continue
        score = 0.0
        score += attr_weight * jaccard_similarity(answer, attr_target)
        score += emo_weight * jaccard_similarity(answer, emo_target)
        score += act_weight * jaccard_similarity(answer, act_target)
        rewards.append(max(0.0, min(1.0, score)))
    return rewards


def accuracy_reward(completions, solution, **kwargs):
    def normalize_number(num_str):
        try:
            return float(num_str.replace(",", ""))
        except Exception:
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        rows, cols = len(ref_words) + 1, len(hyp_words) + 1
        dp = [[0] * cols for _ in range(rows)]
        for row in range(rows):
            dp[row][0] = row
        for col in range(cols):
            dp[0][col] = col
        for row in range(1, rows):
            for col in range(1, cols):
                if ref_words[row - 1] == hyp_words[col - 1]:
                    dp[row][col] = dp[row - 1][col - 1]
                else:
                    dp[row][col] = 1 + min(dp[row - 1][col], dp[row][col - 1], dp[row - 1][col - 1])
        return dp[-1][-1] / max(1, len(ref_words))

    question_type = kwargs["problem_type"][0]
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for content, sol in zip(contents, solution):
        output_ans = extract_answer(content)
        gt_ans = extract_answer(sol)
        if question_type == "multiple choice":
            reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            reward = 1.0 if gt_number is not None and out_number is not None and round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "OCR":
            reward = max(0.0, min(1.0, 1 - wer(gt_ans, output_ans)))
        elif question_type == "free-form":
            scores = scorer.score(gt_ans, output_ans)
            reward = max(0.0, min(1.0, (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3))
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                reward = 0.0
            else:
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                reward = 1 - min(1.0, max(0.0, rel_diff))
        else:
            reward = 0.0
        rewards.append(reward)
        if str(__import__("os").getenv("DEBUG_MODE", "false")).lower() == "true":
            log_path = __import__("os").getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as file:
                    file.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    file.write(f"Content: {content}\nSolution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.fullmatch(pattern, content, re.DOTALL) else 0.0 for content in completion_contents]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "facial_reward": facial_reward,
}


def make_conversation(example, script_args: FaVChatGRPOScriptArguments):
    if example["problem_type"] == "multiple choice":
        question = example["problem"] + "Options:\n" + "\n".join(example.get("options", [])) + "\n"
    else:
        question = example["problem"]

    payload = {
        "prompt": [{
            "role": "user",
            "content": [
                {"type": example["data_type"]},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example["problem_type"]]},
            ],
        }],
        "data_root": script_args.data_root,
        "attr_weight": script_args.attr_weight,
        "emo_weight": script_args.emo_weight,
        "act_weight": script_args.act_weight,
    }
    for key in [
        "path",
        "problem_type",
        "problem_id",
        "data_type",
        "solution",
        "facial_attributes",
        "facial_emotion",
        "facial_action",
        "options",
    ]:
        if key in example:
            payload[key] = example[key]
    return payload


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[name] for name in script_args.reward_funcs]
    if script_args.dataset_name.endswith((".json", ".jsonl")):
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.map(lambda example: make_conversation(example, script_args))

    trainer_cls = FaVChatGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((FaVChatGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
