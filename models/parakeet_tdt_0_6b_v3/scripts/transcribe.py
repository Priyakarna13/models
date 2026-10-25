"""Transcribe 16 kHz mono audio using the complete extracted model package.

Usage: python transcribe.py recording.wav [--package DIRECTORY] [--language auto]
The encoder uses native BF16 tensors; cached states stay in their device buffers.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def log_mel(audio, filters, window, *, center=True, normalize=False):
    from scipy.fft import rfft

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 2:
        raise ValueError("Audio window must contain at least two samples")
    emphasized = np.concatenate((audio[:1], audio[1:] - np.float32(0.97) * audio[:-1]))
    if center:
        emphasized = np.pad(emphasized, (256, 256))
    if len(emphasized) < 512:
        raise ValueError("Audio window is shorter than the Fourier transform")
    frames = np.lib.stride_tricks.sliding_window_view(emphasized, 512)[::160]
    padded_window = np.pad(window, (56, 56))
    spectrum = rfft(frames * padded_window, axis=-1)
    magnitudes = np.sqrt(spectrum.real**2 + spectrum.imag**2) ** 2
    features = np.log(magnitudes @ filters.T + np.float32(2**-24))
    valid = len(audio) // 160 if center else len(features)
    features = features[:valid]
    if normalize:
        if valid < 2:
            raise ValueError("Normalized audio needs at least two feature frames")
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, ddof=1, keepdims=True)
        features = (features - mean) / (std + np.float32(1e-5))
    return features.astype(np.float32)[None]


def bf16_bits(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


class Decoder:
    """Greedy RNNT/TDT with prediction caching and duration carry between chunks."""

    def __init__(self, predictor, joiner, config):
        self.predictor, self.joiner, self.config = predictor, joiner, config
        self.tokens = []
        self.hidden = np.zeros((2, 1, 640), np.float32)
        self.cell = self.hidden.copy()
        self.skip = 0
        self.predict(config["blank_token_id"])

    def predict(self, token):
        self.prediction, self.hidden, self.cell = self.predictor.run(
            None, dict(token=np.array([[token]], np.int64), hidden=self.hidden, cell=self.cell)
        )

    def step(self, encoded):
        frame = self.skip
        while frame < encoded.shape[1]:
            advance = 1
            for _ in range(self.config["max_symbols_per_step"]):
                values = self.joiner.run(
                    None, dict(encoder_frame=encoded[:, frame : frame + 1], predictor_output=self.prediction)
                )
                token = int(values[0].argmax())
                duration = int(values[1].argmax()) if self.config["kind"] == "parakeet" else 0
                if token != self.config["blank_token_id"]:
                    self.tokens.append(token)
                    self.predict(token)
                if self.config["kind"] == "parakeet":
                    if duration or token == self.config["blank_token_id"]:
                        advance = max(1, duration)
                        break
                elif token == self.config["blank_token_id"]:
                    break
            frame += advance
        self.skip = max(0, frame - encoded.shape[1])


class Transcriber:
    """Own one encoder Run. Call transcribe() for each independent utterance."""

    def __init__(self, package):
        import onnxruntime as ort
        import pyneat as pn
        from tokenizers import Tokenizer

        self.pn = pn
        package = Path(package)
        self.config = json.loads((package / "runtime_config.json").read_text())
        self.processor = json.loads((package / "processor_config.json").read_text())
        with np.load(package / "frontend.npz") as data:
            self.filters, self.window = data["mel_filters"], data["window"]
        self.tokenizer = Tokenizer.from_file(str(package / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.predictor, self.joiner = [
            ort.InferenceSession(str(package / (name + "_cpu_int8.onnx")), options, providers=["CPUExecutionProvider"])
            for name in ("predictor", "joiner")
        ]
        archives = list(package.glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise ValueError("Extract one complete model package into its own directory")
        options = pn.ModelOptions()
        terminal = pn.InferenceTerminalPolicy()
        terminal.mla_only = True
        options.inference_terminal = terminal
        self.model = pn.Model(archives[0], options)
        self.graph = pn.Graph("speech_encoder")
        ingress = pn.InputOptions()
        ingress.payload_type = pn.PayloadType.Tensor
        ingress.format = pn.Format.BF16
        self.graph.add(pn.nodes.input("features", ingress))
        self.graph.add(self.model.inference())
        self.graph.add(pn.nodes.output("encoded"))
        self.run = None
        # Many logical caches share one immutable zero allocation on the first call.
        # The Sample API copies tensor descriptors while retaining their shared storage.
        self.zero_samples = []
        total = sum((port["bytes"] + 4095) // 4096 * 4096 for port in self.config["inputs"])
        arena = pn.Tensor.from_numpy(np.zeros(total, np.uint8), copy=True, memory=pn.TensorMemory.EV74)
        self.zero_states = {}
        offset = 0
        for port in self.config["inputs"]:
            holder = pn.Sample()
            holder.tensor = arena
            tensor = holder.tensor
            tensor.dtype = pn.TensorDType.BFloat16
            tensor.shape = port["shape"]
            tensor.strides_bytes = []
            tensor.byte_offset = offset
            self.zero_states[port["name"]] = tensor
            self.zero_samples.append(holder)
            offset += (port["bytes"] + 4095) // 4096 * 4096

    def close(self):
        if self.run is not None:
            self.run.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def tensor(self, value):
        tensor = self.pn.Tensor.from_numpy(bf16_bits(value), copy=True, memory=self.pn.TensorMemory.EV74)
        tensor.dtype = self.pn.TensorDType.BFloat16
        return tensor

    def encode(self, inputs):
        ordered = [inputs[port["name"]] for port in self.config["inputs"]]
        if self.run is None:
            options = self.pn.RunOptions()
            options.output_memory = self.pn.OutputMemory.ZeroCopy
            options.startup_preflight = False
            self.run = self.graph.build(ordered, options)
        tensors = self.run.run(ordered, 60000)
        if len(tensors) != len(self.config["outputs"]):
            raise RuntimeError(f"Encoder returned {len(tensors)} tensors; expected {len(self.config['outputs'])}")
        outputs = dict(zip((port["name"] for port in self.config["outputs"]), tensors, strict=True))
        holder = self.pn.Sample()
        holder.tensor = outputs["encoded"]
        words = holder.tensor
        words.dtype = self.pn.TensorDType.UInt16
        encoded = (words.to_numpy().astype(np.uint32) << 16).view(np.float32).copy()
        if not np.isfinite(encoded).all():
            raise RuntimeError("Encoder returned nonfinite features")
        return encoded, outputs

    def transcribe(self, audio, language="auto"):
        """Yield cumulative text after each 560 ms chunk or 2 s central window."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono PCM samples")
        decoder = Decoder(self.predictor, self.joiner, self.config)
        states = dict(self.zero_states)
        if self.config["kind"] == "nemotron":
            prompt = self.processor["prompt_dictionary"].get(language)
            if prompt is None:
                raise ValueError(f"Unknown language prompt: {language}")
            one_hot = np.zeros((1, 1, 128), np.float32)
            one_hot[..., prompt] = 1
            states["language_one_hot"] = self.tensor(one_hot)
            index = mel_index = 0
            while len(audio):
                first = index == 0
                start = 0 if first else mel_index * 160 - 256
                if start >= len(audio):
                    break
                size = 7880 if first else 9360
                real = min(size, len(audio) - start)
                raw = np.pad(audio[start : start + size], (0, size - real))
                features = log_mel(raw, self.filters, self.window, center=first)
                count = 49 if first else 56
                features = features[:, :count]
                fixed = np.pad(features, ((0, 0), (7 if first else 0, 0), (0, 0)))
                mask = np.ones((1, 48, 1, 1), np.float32)
                if first:
                    mask[:, :3] = 0
                    mask[:, 32] = 0
                bias = np.zeros((1, 7, 64, 1), np.float32)
                bias[:, :, : 56 - min(index * 7, 56)] = -10000
                states.update(
                    features=self.tensor(fixed), subsampling_mask=self.tensor(mask), attention_bias=self.tensor(bias)
                )
                encoded, outputs = self.encode(states)
                states["sub0"] = self.tensor(
                    np.pad(features[:, -1:, None, :], ((0, 0), (0, 0), (0, 0), (2, 6))).transpose(0, 2, 3, 1)
                )
                states["sub1"], states["sub2"] = outputs["sub/next_sub1"], outputs["sub/next_sub2"]
                for layer in range(24):
                    for name in ("keys", "values", "conv"):
                        states[f"b{layer:02d}/{name}"] = outputs[f"b{layer:02d}/next_{name}"]
                decoder.step(encoded)
                yield self.tokenizer.decode(decoder.tokens, skip_special_tokens=True)
                mel_index += count
                index += 1
                if real < size:
                    break
        else:
            if language != "auto":
                raise ValueError("Parakeet has no language prompt")
            for target in range(0, len(audio), 32000):
                start, end = max(0, target - 160000), min(len(audio), target + 64000)
                raw = audio[start:end]
                if len(raw) < 320:
                    raw = np.pad(raw, (0, 320 - len(raw)))
                features = log_mel(raw, self.filters, self.window, normalize=True)
                valid = min(1400, features.shape[1])
                fixed = np.pad(features[:, :valid], ((0, 0), (0, 1400 - valid), (0, 0)))[..., None]
                masks = np.zeros((1, 1232, 1, 1), np.float32)
                for stride, offset, size in zip((2, 4, 8), (0, 704, 1056), (700, 350, 175), strict=True):
                    masks[:, offset : offset + min((valid + stride - 1) // stride, size)] = 1
                bias = np.zeros((1, 1, 176, 1), np.float32)
                bias[:, :, (valid + 7) // 8 :] = -10000
                encoded, _ = self.encode(
                    dict(
                        features=self.tensor(fixed), padding_masks=self.tensor(masks), attention_bias=self.tensor(bias)
                    )
                )
                first = (target - start) // 1280
                count = min(25, (len(audio) - target + 1279) // 1280)
                decoder.step(encoded[:, first : first + count])
                yield self.tokenizer.decode(decoder.tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()
    with wave.open(str(args.audio), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            parser.error("Expected 16 kHz mono PCM16 WAV")
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
    with Transcriber(args.package) as transcriber:
        for text in transcriber.transcribe(audio, args.language):
            print(text, flush=True)


if __name__ == "__main__":
    main()
