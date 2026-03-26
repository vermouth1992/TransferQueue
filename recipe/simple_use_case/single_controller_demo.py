# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

import transfer_queue as tq
from transfer_queue import KVBatchMeta

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"
ray.init()


def compute_log_prob(data1, _data2):
    time.sleep(3)
    return data1


def compute_loss(data1, _data2):
    time.sleep(3)
    return data1


def generate_sequences(data):
    time.sleep(3)
    return data


class TrainingWorker:
    def __init__(self, role):
        self.role = role

    def train_mini_batch(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        """Simulate multi-mini-batch training loop"""

        assert self.role == "actor"

        # 1. Pull data from storage
        data = tq.kv_batch_get_by_meta(meta=kv_meta)
        logger.info(f"train_mini_batch: got data {data}")

        # 2. Compute loss
        output = compute_loss(data["old_log_prob"], data["ref_log_prob"])
        output = TensorDict({"loss": output}, batch_size=output.size(0))

        # 3. Write back
        kv_meta = tq.kv_batch_put(keys=kv_meta.keys, partition_id=kv_meta.partition_id, fields=output)
        logger.info("train_mini_batch: put data done")

        return kv_meta

    def infer_batch(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        """Simulate forward-only inference"""
        # 1. Pull data from storage
        data = tq.kv_batch_get_by_meta(meta=kv_meta)
        logger.info(f"compute_log_prob: got data {data}")

        # 2. Model forward
        output = compute_log_prob(data["input_ids"], data["generate_sequences_ids"])
        if self.role == "actor":
            output = TensorDict({"old_log_prob": output}, batch_size=output.size(0))
        elif self.role == "ref":
            output = TensorDict({"ref_log_prob": output}, batch_size=output.size(0))
        else:
            raise ValueError(f"Role {self.role} not supported.")

        # 3. Write back
        kv_meta = tq.kv_batch_put(keys=kv_meta.keys, partition_id=kv_meta.partition_id, fields=output)
        logger.info("infer_batch: put data done")

        return kv_meta


class ActorRolloutRefWorker:
    def __init__(self):
        self.actor = TrainingWorker(role="actor")
        self.ref = TrainingWorker(role="ref")

    def compute_ref_log_prob(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        output = self.ref.infer_batch(kv_meta)
        return output

    def compute_log_prob(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        output = self.actor.infer_batch(kv_meta)
        return output

    def update_actor(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        output = self.actor.train_mini_batch(kv_meta)
        return output

    async def update_weights(self, global_steps: int = None):
        # Simulate weight sync from actor to rollout
        logger.info(f"update_weights: syncing weights at step {global_steps}")
        await asyncio.sleep(1)


@ray.remote
class AsyncvLLMServer:
    def __init__(self, config):
        tq.init(config)

    async def generate(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        data = tq.kv_batch_get_by_meta(meta=kv_meta)
        logger.info(f"demo get data -> generate_sequences {data}")

        data = data["input_ids"]
        data += 1
        await asyncio.sleep(3)

        output = TensorDict(
            {
                "generate_sequences_ids": data,
                "non_tensor_data": torch.stack([NonTensorData("test_str") for _ in range(data.size(0))]),
                "nested_tensor": torch.nested.as_nested_tensor(
                    [torch.randn(1, 2) for _ in range(data.size(0))], layout=torch.jagged
                ),
            },
            batch_size=data.size(0),
        )

        kv_meta = tq.kv_batch_put(keys=kv_meta.keys, partition_id=kv_meta.partition_id, fields=output)
        logger.info("demo Async Server put data to storages done")

        return kv_meta



async def generate(prompt: torch.Tensor, response_length: int, vocab_size: int) -> torch.Tensor:
    assert prompt.ndim == 1
    response = torch.randint(low=0, high=vocab_size, size=(response_length,), dtype=torch.long)
    return response



IMAGE_TOKEN_ID = 32001


def simulate_chat_template(
    messages: list[dict], vocab_size: int, image_token_length: int = 64
) -> torch.Tensor:
    """Simulate ``tokenizer.apply_chat_template`` with interleaved image support.

    Each message follows the OpenAI-style multi-modal format::

        {"role": "user",
         "content": [
             {"type": "image_url", "image_url": {"url": "..."}},
             {"type": "text", "text": "Describe this image"},
         ]}

    ``content`` may also be a plain string for text-only messages.

    - ``"text"`` parts are tokenised as one random ID per whitespace word.
    - ``"image_url"`` parts each produce ``image_token_length`` placeholder
      tokens (simulating the patch embeddings a vision encoder would emit).

    Args:
        messages: Chat-style message list.
        vocab_size: Vocabulary size for random text token IDs.
        image_token_length: Number of placeholder tokens per image.

    Returns:
        1-D ``torch.Tensor`` of token IDs.
    """
    tokens: list[int] = []
    for msg in messages:
        content = msg.get("content", "")

        if isinstance(content, str):
            if content:
                tokens.extend(torch.randint(0, vocab_size, (len(content.split()),)).tolist())
        elif isinstance(content, list):
            for part in content:
                part_type = part.get("type", "")
                if part_type == "text":
                    text = part.get("text", "")
                    if text:
                        tokens.extend(torch.randint(0, vocab_size, (len(text.split()),)).tolist())
                elif part_type == "image_url":
                    tokens.extend([IMAGE_TOKEN_ID] * image_token_length)

    return torch.tensor(tokens, dtype=torch.long)


@dataclass
class AgentLoopConfig:
    """Configuration for :class:`AgentLoop` multi-turn rollout."""

    max_turns_range: tuple[int, int] = (1, 4)
    tool_response_length_range: tuple[int, int] = (5, 20)
    vocab_size: int = 32000
    response_length: int = 32
    image_token_length: int = 64


class AgentLoop:
    """Multi-turn agentic rollout that interleaves LLM generation with tool calls.

    Each turn:
      1. Call ``generate()`` to produce a model response.
      2. Check whether the response triggers a tool call.
      3. If yes, simulate tool execution and append the tool-response tokens.
      4. Repeat until no tool call is detected or ``max_turns`` is reached.
    """

    def __init__(self, config: AgentLoopConfig):
        self.config = config

    async def run(self, messages: list[dict]) -> torch.Tensor:
        """Execute a multi-turn rollout starting from *messages*.

        Args:
            messages: OpenAI-style multi-modal message list.  ``content``
                is a list of typed parts (``"text"`` / ``"image_url"``).
                Example::

                    [{"role": "user",
                      "content": [
                          {"type": "image_url",
                           "image_url": {"url": "https://...jpg"}},
                          {"type": "text",
                           "text": "Describe this image"},
                      ]}]

        Returns:
            1-D ``torch.Tensor`` of token IDs for the full conversation
            (prompt + all generation turns + tool responses).
        """
        cfg = self.config
        min_turns, max_turns = cfg.max_turns_range
        num_turns = random.randint(min_turns, max_turns)

        conversation = simulate_chat_template(messages, cfg.vocab_size, cfg.image_token_length)
        logger.info(
            f"AgentLoop: initial prompt length = {conversation.shape[0]}, "
            f"sampled {num_turns} turns (range {cfg.max_turns_range})"
        )

        for turn in range(num_turns):
            response = await generate(conversation, cfg.response_length, cfg.vocab_size)
            conversation = torch.cat([conversation, response])
            logger.info(
                f"AgentLoop turn {turn}/{num_turns}: generated {response.shape[0]} tokens, "
                f"conversation length = {conversation.shape[0]}"
            )

            if not self._detect_tool_call(turn, num_turns):
                logger.info(f"AgentLoop turn {turn}: final answer produced, rollout complete.")
                break

            tool_response = self._simulate_tool_response()
            conversation = torch.cat([conversation, tool_response])
            logger.info(
                f"AgentLoop turn {turn}: tool call → appended {tool_response.shape[0]} "
                f"tool-response tokens, conversation length = {conversation.shape[0]}"
            )

        return conversation

    def _detect_tool_call(self, turn: int, num_turns: int) -> bool:
        """Simulate tool-call detection.

        In a real agent this would parse the decoded model output for
        tool-call syntax (e.g. function-call JSON).  Here we
        deterministically issue a tool call on every turn except the last
        one, guaranteeing multi-turn behaviour in the demo.
        """
        return turn < num_turns - 1

    def _simulate_tool_response(self) -> torch.Tensor:
        """Simulate tool execution returning random token IDs.

        The response length is sampled uniformly from
        ``[tool_response_length_range[0], tool_response_length_range[1]]``.
        """
        min_len, max_len = self.config.tool_response_length_range
        length = random.randint(min_len, max_len)
        return torch.randint(0, self.config.vocab_size, (length,), dtype=torch.long)




@ray.remote(num_cpus=1)
class AgentLoopWorker:
    def __init__(self, config):
        self.async_vllm_server = AsyncvLLMServer.remote(config)

    async def generate_sequences(self, kv_meta_chunk):
        if isinstance(kv_meta_chunk, list):
            tasks = []
            for item in kv_meta_chunk:
                # asyncio.create_task cannot directly call Ray Actor methods,
                # otherwise an error will be reported：a coroutine was expected, got ObjectRef(xxx)
                tasks.append(asyncio.create_task(self.generate(item)))
            kv_metas = await asyncio.gather(*tasks)
            return KVBatchMeta.concat(kv_metas)

        elif isinstance(kv_meta_chunk, KVBatchMeta):
            kv_meta = await self.generate(kv_meta_chunk)
            return kv_meta

        else:
            raise TypeError(f"Unsupported type for kv_meta_chunk: {type(kv_meta_chunk)}")

    async def generate(self, kv_meta):
        kv_meta_new = await self.async_vllm_server.generate.remote(kv_meta)
        return kv_meta_new


class AgentLoopManager:
    def __init__(self, config):
        self.config = config
        tq.init(config)

        self.async_rollout_workers = []
        num_workers = self.config.rollout_agent_num_workers

        for _ in range(num_workers):
            self.async_rollout_workers.append(AgentLoopWorker.remote(config))

    def generate_sequences(self, kv_meta):
        kv_meta_chunks = kv_meta.chunk(len(self.async_rollout_workers))
        kv_metas = ray.get(
            [
                worker.generate_sequences.remote(kv_meta_chunk)
                for worker, kv_meta_chunk in zip(self.async_rollout_workers, kv_meta_chunks, strict=True)
            ]
        )
        kv_meta = KVBatchMeta.concat(kv_metas)
        logger.info(f"KVBatchMeta: {kv_meta}")

        return kv_meta


class Trainer:
    def __init__(self, config):
        self.config = config
        tq.init(config)
        self.tq_client = tq.get_client()
        self.actor_rollout_wg = ActorRolloutRefWorker()
        self.async_rollout_manager = AgentLoopManager(self.config)

    def fit(self):
        for _epoch in range(1):
            train_dataloader = 1
            for step in range(train_dataloader):
                # ========================= Construct prompt batch data =========================
                input_ids = (
                    torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [100, 111], [200, 222], [300, 333]])
                ) * (step + 1)
                input_ids_repeated = torch.repeat_interleave(input_ids, self.config.num_n_samples, dim=0)
                batch_keys = [str(uuid.uuid4()) for _ in range(len(input_ids_repeated))]
                prompt_batch = TensorDict(
                    {"input_ids": input_ids_repeated, "attention_mask": input_ids_repeated},
                    batch_size=input_ids_repeated.size(0),
                )

                # ========================= Put prompts to TQ system =========================
                tq.kv_batch_put(keys=batch_keys, partition_id=f"train_{step}", fields=prompt_batch)
                logger.info("demo put prompts ok! ")
                time.sleep(5)

                # ========================= Sample generate KVBatchMeta =========================
                sampled_keys = random.sample(batch_keys, self.config.global_batch_size)
                meta = KVBatchMeta(
                    keys=sampled_keys,
                    tags=[{} for _ in sampled_keys],
                    partition_id=f"train_{step}",
                    fields=["input_ids", "attention_mask"],
                )
                logger.info(f"demo get KVBatchMeta {meta}")

                # ========================= Rollout: generate sequences =========================
                meta = self.async_rollout_manager.generate_sequences(meta)
                logger.info(f"demo get after gen KVBatchMeta {meta}")

                # ========================= Compute ref log prob =========================
                meta.fields = ["input_ids", "attention_mask", "generate_sequences_ids"]
                meta = self.actor_rollout_wg.compute_ref_log_prob(meta)
                logger.info(f"demo get ref log prob KVBatchMeta: {meta}")

                # ========================= Compute old log prob =========================
                meta.fields = ["input_ids", "attention_mask", "generate_sequences_ids"]
                meta = self.actor_rollout_wg.compute_log_prob(meta)
                logger.info(f"demo get old log prob KVBatchMeta: {meta}")

                # ========================= Compute reward =========================
                # Simulated inline; in real training this calls a reward model worker
                meta.fields = ["generate_sequences_ids", "ref_log_prob", "old_log_prob"]
                logger.info("demo computing reward (simulated)")
                time.sleep(1)
                logger.info(f"demo reward KVBatchMeta: {meta}")

                # ========================= Update actor =========================
                meta.fields = [
                    "input_ids",
                    "attention_mask",
                    "generate_sequences_ids",
                    "old_log_prob",
                    "ref_log_prob",
                ]
                meta = self.actor_rollout_wg.update_actor(meta)
                logger.info(f"demo get after update actor KVBatchMeta: {meta}")

                # ========================= Sync weights to rollout =========================
                asyncio.run(self.actor_rollout_wg.update_weights(global_steps=step))
                logger.info("demo update weights done")

                # ========================= Clear partition in TQ =========================
                self.tq_client.clear_partition(partition_id=f"train_{step}")
                logger.info("clear ok! ")
        logger.info("demo done!")

        # Cleanup resources
        self.tq_client.close()


if __name__ == "__main__":
    # Demo-level training hyperparameters (not part of TQ config)
    demo_conf = OmegaConf.create(
        {
            "global_batch_size": 8,
            "num_global_batch": 1,
            "rollout_agent_num_workers": 2,
            "num_n_samples": 2,
            # AgentLoop multi-turn rollout settings
            "max_turns_range": [1, 4],
            "tool_response_length_range": [5, 20],
            "vocab_size": 32000,
            "response_length": 32,
            "image_token_length": 64,
        }
    )

    # Load default TQ config and override as needed
    tq_conf = OmegaConf.load(resources.files("transfer_queue") / "config.yaml")
    tq_conf = OmegaConf.merge(tq_conf, {"backend": {"SimpleStorage": {"num_data_storage_units": 2}}})

    dict_conf = OmegaConf.merge(demo_conf, tq_conf)

    trainer = Trainer(dict_conf)
    trainer.fit()

    ray.shutdown()
