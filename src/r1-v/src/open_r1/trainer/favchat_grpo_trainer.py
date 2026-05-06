from __future__ import annotations

import copy
import os
from typing import Any, Union

import torch
from transformers import PreTrainedModel

from qwen_vl_utils import process_vision_info
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation

from open_r1.favchat_components import (
    FaVChatPromptQueryEncoder,
    SampleValueBaseline,
    build_feature_pyramid,
    pairwise_reward_statistics,
)
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer


class FaVChatGRPOTrainer(Qwen2VLGRPOTrainer):
    def __init__(self, *args, script_args=None, **kwargs):
        self.use_de_grpo = getattr(script_args, "use_de_grpo", True)
        self.remove_threshold = getattr(script_args, "remove_threshold", 0.20)
        self.keep_threshold = getattr(script_args, "keep_threshold", 0.80)
        self.utility_smoothing = getattr(script_args, "utility_smoothing", 0.80)
        self.reward_decay = getattr(script_args, "reward_decay", 0.50)
        self.sample_state = {}
        super().__init__(*args, script_args=script_args, **kwargs)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", 3584)
        hidden_size = hidden_size or 3584
        num_heads = max(1, min(8, hidden_size // 128))
        self.model.favchat_encoder = FaVChatPromptQueryEncoder(hidden_size=hidden_size, num_heads=num_heads)
        self.model.favchat_value_head = SampleValueBaseline(hidden_size=hidden_size)
        model_device = next(self.model.parameters()).device
        self.model.favchat_encoder.to(model_device)
        self.model.favchat_value_head.to(model_device)

    def _resolve_multimodal_inputs(self, inputs):
        input_copy = copy.deepcopy(inputs[0]["prompt"])
        input_copy = self.remove_none_from_data(input_copy)
        data_root = inputs[0].get("data_root", "./FaVChat-data")
        path = inputs[0]["path"]
        media_path = os.path.join(os.getcwd(), data_root.lstrip("./"), path.lstrip("/"))
        if inputs[0]["data_type"] == "image":
            input_copy[0]["content"][0]["image"] = media_path
        elif inputs[0]["data_type"] == "video":
            input_copy[0]["content"][0]["video"] = media_path
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        except Exception as error:
            raise RuntimeError(f"process_vision_info failed for {media_path}: {error}") from error
        return input_copy, image_inputs, video_inputs, video_kwargs

    def _compute_favchat_state(self, model, prompt_inputs):
        hidden_outputs = model(**prompt_inputs, output_hidden_states=True)
        hidden_states = hidden_outputs.hidden_states
        text_tokens = hidden_states[-1]
        general_tokens = hidden_states[-1]
        facial_pyramid = build_feature_pyramid(hidden_states, num_layers=4)
        favchat_state = model.favchat_encoder(
            text_tokens=text_tokens,
            general_tokens=general_tokens,
            facial_feature_pyramid=facial_pyramid,
        )
        sample_embedding = favchat_state.fused_tokens.mean(dim=1)
        sample_baseline = model.favchat_value_head(sample_embedding)
        return favchat_state, sample_embedding, sample_baseline

    def _compute_de_grpo_advantages(self, rewards, completion_mask, per_token_logps, sample_baseline):
        standard_mean = rewards.view(-1, self.num_generations).mean(dim=1)
        standard_std = rewards.view(-1, self.num_generations).std(dim=1)
        repeated_mean = standard_mean.repeat_interleave(self.num_generations, dim=0)
        repeated_std = standard_std.repeat_interleave(self.num_generations, dim=0)
        standard_advantages = (rewards - repeated_mean) / (repeated_std + 1e-4)

        response_scores = ((per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).detach()
        sample_reward_margin, sample_reward_gap, _ = pairwise_reward_statistics(rewards.detach(), self.num_generations)
        _, sample_gradient_gap, _ = pairwise_reward_statistics(response_scores, self.num_generations)
        sample_utility = sample_reward_gap * sample_gradient_gap

        lifecycle = []
        sample_ids = getattr(self, "_latest_problem_ids", [str(index) for index in range(sample_utility.size(0))])
        baseline = sample_baseline.float()
        signed_margin = sample_reward_margin.float()
        for index, sample_id in enumerate(sample_ids):
            previous = float(self.sample_state.get(sample_id, 0.0))
            current = float(sample_utility[index].detach().item())
            smoothed = self.utility_smoothing * previous + (1 - self.utility_smoothing) * current
            self.sample_state[sample_id] = smoothed
            if smoothed < self.remove_threshold:
                delta = 0.0
            elif smoothed > self.keep_threshold:
                delta = self.reward_decay
            else:
                delta = 1.0
            lifecycle.append(delta)
        lifecycle = rewards.new_tensor(lifecycle)
        sample_advantages = lifecycle * (signed_margin - baseline)
        repeated_sample_advantages = sample_advantages.repeat_interleave(self.num_generations, dim=0)
        if self.use_de_grpo:
            final_advantages = standard_advantages * torch.tanh(repeated_sample_advantages)
        else:
            final_advantages = standard_advantages
        metrics = {
            "sample_margin": signed_margin.mean().item(),
            "sample_utility": sample_utility.mean().item(),
            "sample_baseline": baseline.mean().item(),
            "sample_lifecycle": lifecycle.mean().item(),
        }
        return final_advantages, repeated_std, metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [example["prompt"] for example in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        self._latest_problem_ids = [str(example.get("problem_id", idx)) for idx, example in enumerate(inputs)]
        input_copy, image_inputs, video_inputs, _ = self._resolve_multimodal_inputs(inputs)

        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        favchat_state, _, sample_baseline = self._compute_favchat_state(model, prompt_inputs)

        shuffled_rewards_per_func = None
        if self.temporal and video_inputs:
            indices = torch.randperm(video_inputs[0].size(0))
            shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            if self.max_prompt_length is not None:
                shuffled_prompt_inputs["input_ids"] = shuffled_prompt_inputs["input_ids"][:, -self.max_prompt_length :]
                shuffled_prompt_inputs["attention_mask"] = shuffled_prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            if self.temporal and video_inputs:
                shuffled_prompt_completion_ids = unwrapped_model.generate(
                    **shuffled_prompt_inputs,
                    generation_config=self.shuffled_generation_config,
                )
                shuffled_prompt_length = shuffled_prompt_inputs["input_ids"].size(1)
                shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_inputs_for_logps = {key: value for key, value in prompt_inputs.items() if key not in {"input_ids", "attention_mask"}}
        if inputs[0]["data_type"] == "image":
            repeat_shape = [len(prompt_completion_ids)] + [1] * (prompt_inputs_for_logps["pixel_values"].dim() - 1)
            prompt_inputs_for_logps["pixel_values"] = prompt_inputs_for_logps["pixel_values"].repeat(*repeat_shape)
            prompt_inputs_for_logps["image_grid_thw"] = prompt_inputs_for_logps["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        if inputs[0]["data_type"] == "video":
            repeat_shape = [len(prompt_completion_ids)] + [1] * (prompt_inputs_for_logps["pixel_values_videos"].dim() - 1)
            prompt_inputs_for_logps["pixel_values_videos"] = prompt_inputs_for_logps["pixel_values_videos"].repeat(*repeat_shape)
            prompt_inputs_for_logps["video_grid_thw"] = prompt_inputs_for_logps["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if "second_per_grid_ts" in prompt_inputs_for_logps:
                del prompt_inputs_for_logps["second_per_grid_ts"]

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs_for_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs_for_logps)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs_for_logps)
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        if self.temporal and video_inputs:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": item}] for item in shuffled_completions]
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for index, reward_func in enumerate(self.reward_funcs):
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                reward_values = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **reward_kwargs)
                shuffled_rewards_per_func[:, index] = torch.tensor(reward_values, dtype=torch.float32, device=device)

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": item}] for item in completions]
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for index, reward_func in enumerate(self.reward_funcs):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            reward_values = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, index] = torch.tensor(reward_values, dtype=torch.float32, device=device)

        if self.temporal and video_inputs:
            temporal_rewards_per_func = rewards_per_func.clone()
            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()
            if acc_mean >= 0.8 * shuffled_acc_mean:
                temporal_rewards_per_func[temporal_rewards_per_func[:, 0] > 0.1, 0] += 0.3
                temporal_rewards = torch.tensor([1.0], device=device)
            else:
                temporal_rewards = torch.tensor([0.0], device=device)
            rewards = temporal_rewards_per_func.sum(dim=1)
        else:
            temporal_rewards = torch.tensor([0.5], device=device)
            rewards = rewards_per_func.sum(dim=1)

        if self.len_control:
            valid_indices = torch.nonzero(rewards_per_func[:, 0] > 0.1, as_tuple=True)[0].tolist()
            lengths = completion_mask.sum(1)
            if len(valid_indices) > 1:
                for index in valid_indices:
                    if 320 <= lengths[index] <= 512:
                        rewards[index] += 0.2

        advantages, std_grouped_rewards, de_metrics = self._compute_de_grpo_advantages(
            rewards=rewards,
            completion_mask=completion_mask,
            per_token_logps=per_token_logps,
            sample_baseline=sample_baseline,
        )

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for index, reward_func in enumerate(self.reward_funcs):
            reward_name = reward_func.config._name_or_path.split("/")[-1] if isinstance(reward_func, PreTrainedModel) else reward_func.__name__
            self._metrics[f"rewards/{reward_name}"].append(reward_per_func[index].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()).mean().item()
        )
        self._metrics["temporal_rewards"].append(self.accelerator.gather_for_metrics(temporal_rewards).mean().item())
        self._metrics["favchat/general_weight"].append(favchat_state.general_weight.mean().item())
        self._metrics["favchat/facial_weight"].append(favchat_state.facial_weight.mean().item())
        for key, value in de_metrics.items():
            self._metrics[f"de_grpo/{key}"].append(value)
        return loss
