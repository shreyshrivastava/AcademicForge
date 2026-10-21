"""Quantized FLAN-T5 on CPU, with one cached instance and bounded generation.

Public, pinned ONNX artifacts; no token, Torch, remote Python, or paid API.
Loading happens in a background thread so paper search remains usable.
"""

from functools import lru_cache
import os
from threading import Lock, Thread
import time

MODEL_ID = "google/flan-t5-small"
ARTIFACT_REPO = "onnx-community/flan-t5-small-ONNX"
REVISION = "76988c16f73cadb2c2e13e2d7d85608944223105"


class SmallModel:
    def __init__(self):
        self.state = "not_started"
        self.failure = ""
        self.lock = Lock()
        self.inference = Lock()
        self.encoder = self.decoder = self.tokenizer = None

    def status(self, start=True):
        if os.getenv("ACADEMICFORGE_DEMO_MODEL_ENABLED", "true").lower() != "true":
            return "disabled"
        with self.lock:
            if start and self.state == "not_started":
                self.state = "loading"
                Thread(target=self._load, daemon=True, name="academicforge-small-model").start()
            return self.state

    def _load(self):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            from huggingface_hub import hf_hub_download

            def artifact(name):
                return hf_hub_download(ARTIFACT_REPO, name, revision=REVISION, token=False)

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            encoder = ort.InferenceSession(artifact("onnx/encoder_model_quantized.onnx"), options,
                                           providers=["CPUExecutionProvider"])
            decoder = ort.InferenceSession(artifact("onnx/decoder_model_quantized.onnx"), options,
                                           providers=["CPUExecutionProvider"])
            tokenizer = Tokenizer.from_file(artifact("tokenizer.json"))
            tokenizer.enable_truncation(max_length=512)
            with self.lock:
                self.encoder, self.decoder, self.tokenizer = encoder, decoder, tokenizer
                self.state = "ready"
        except Exception as exc:
            # Do not surface paths, authentication state, or downloader tracebacks to users.
            with self.lock:
                self.state = "unavailable"
                self.failure = type(exc).__name__

    def retry(self):
        with self.lock:
            if self.state == "unavailable":
                self.state = "not_started"
                self.failure = ""

    def generate(self, prompt, max_tokens=96, seconds=15):
        state = self.status()
        if state != "ready":
            return None, f"model_{state}"
        if not self.inference.acquire(blocking=False):
            return None, "model_busy"
        try:
            import numpy as np
            deadline = time.monotonic() + seconds
            encoding = self.tokenizer.encode(prompt)
            ids = np.array([encoding.ids], dtype=np.int64)
            mask = np.array([encoding.attention_mask], dtype=np.int64)
            hidden = self.encoder.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            output = [0]  # T5 decoder start/pad token; EOS is 1.
            for _ in range(min(max_tokens, 128)):
                if time.monotonic() >= deadline:
                    return None, "model_timeout"
                feeds = {"input_ids": np.array([output], dtype=np.int64),
                         "encoder_hidden_states": hidden, "encoder_attention_mask": mask}
                inputs = {item.name for item in self.decoder.get_inputs()}
                logits = self.decoder.run(None, {k: v for k, v in feeds.items() if k in inputs})[0]
                next_token = int(np.argmax(logits[0, -1]))
                if next_token == 1:
                    break
                output.append(next_token)
            text = self.tokenizer.decode(output[1:], skip_special_tokens=True).strip()
            if len(text.split()) < 6:
                return None, "model_output_unusable"
            return text, None
        except Exception:
            return None, "model_generation_failed"
        finally:
            self.inference.release()


@lru_cache(maxsize=1)
def get_model():
    return SmallModel()
