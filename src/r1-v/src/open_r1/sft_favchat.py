from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import ModelConfig, SFTConfig, SFTTrainer, ScriptArguments, TrlParser, get_kbit_device_map, get_peft_config


@dataclass
class FaVChatSFTScriptArguments(ScriptArguments):
    data_root: Optional[str] = field(default="./FaVChat-data", metadata={"help": "Dataset root path."})


processor = None


def get_current_device():
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def prepare_dataset(example: Dict[str, Any], data_root: str) -> Dict[str, List[Dict[str, Any]]]:
    question = example["problem"]
    if example["problem_type"] == "multiple choice":
        question += "Options:\n" + "\n".join(example.get("options", [])) + "\n"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful facial video understanding assistant."}]},
        {
            "role": "user",
            "content": [
                {example["data_type"]: os.path.join(os.getcwd(), data_root.lstrip("./"), example["path"].lstrip("/")), "type": example["data_type"]},
                {"type": "text", "text": question + "\nReason inside <think> and answer inside <answer>."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": example.get("process", "") + "\n" + example["solution"]}]},
    ]
    return {"messages": messages}


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    texts = []
    image_inputs = []
    video_inputs = []
    for index, example in enumerate(examples):
        try:
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            example_images, example_videos, _ = process_vision_info(example["messages"], return_video_kwargs=True)
            if example_images is not None:
                image_inputs.extend(example_images)
            if example_videos is not None:
                video_inputs.extend(example_videos)
        except Exception as error:
            raise ValueError(f"Failed to process example {index}: {error}") from error
    inputs = processor(
        text=texts,
        images=image_inputs or None,
        videos=video_inputs or None,
        return_tensors="pt",
        padding=True,
    )
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100
    inputs["labels"] = labels
    return inputs


if __name__ == "__main__":
    parser = TrlParser((FaVChatSFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    if script_args.dataset_name.endswith((".json", ".jsonl")):
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    torch_dtype = model_config.torch_dtype if model_config.torch_dtype in ["auto", None] else getattr(torch, model_config.torch_dtype)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
    )
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    prepared_dataset = [prepare_dataset(example, script_args.data_root) for example in dataset["train"]]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
