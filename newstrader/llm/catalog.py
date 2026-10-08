"""Exactly what Pro AI downloads: one pinned llama.cpp server build per platform and a short list of models, each with
its size and SHA-256 so a broken or tampered download is rejected. Found with scripts/check_llm_assets.py
(.github/workflows/check-llm-assets.yml); CI re-checks the pins.

All models are Apache-2.0, not gated on Hugging Face, and run with "thinking" switched off (it adds seconds to minutes
per story).
"""

from __future__ import annotations

from dataclasses import dataclass

LLAMA_CPP_BUILD = "b11512"
_RELEASES = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}"


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{_RELEASES}/{self.name}"


@dataclass(frozen=True)
class ServerBuild:
    key: str  # cuda13 / cuda12 / vulkan / cpu / metal / linux-cpu / linux-vulkan
    label: str
    os: str  # windows / mac / linux
    assets: tuple[Asset, ...]  # the server zip first, then any runtime DLL zip unpacked next to it
    min_driver: float = 0.0  # NVIDIA driver version needed (Windows CUDA builds)

    @property
    def download_bytes(self) -> int:
        return sum(a.size for a in self.assets)


SERVER_BUILDS: dict[str, ServerBuild] = {b.key: b for b in (
    ServerBuild("cuda13", "NVIDIA (CUDA 13)", "windows", (
        Asset("llama-b11512-bin-win-cuda-13.4-x64.zip", 153_412_926,
              "998583ef7f12a742ce98dab8611d9d976fe3e0f889d607757e13b5d588663428"),
        Asset("cudart-llama-bin-win-cuda-13.4-x64.zip", 423_535_356,
              "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668"),
    ), min_driver=580.0),
    ServerBuild("cuda12", "NVIDIA (CUDA 12, older drivers)", "windows", (
        Asset("llama-b11512-bin-win-cuda-12.4-x64.zip", 265_174_690,
              "1eecf1115b27ecd947c215426abaed3eb75991eca7aa56663bd74ec7c19ab56d"),
        Asset("cudart-llama-bin-win-cuda-12.4-x64.zip", 391_443_627,
              "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
    ), min_driver=551.61),
    ServerBuild("vulkan", "Any graphics card (Vulkan)", "windows", (
        Asset("llama-b11512-bin-win-vulkan-x64.zip", 33_445_941,
              "e510057785e012b07f582c9647356a9d37c8376334ac2a1fe32a8f8456a6ef52"),
    )),
    ServerBuild("cpu", "Processor only (slow)", "windows", (
        Asset("llama-b11512-bin-win-cpu-x64.zip", 19_484_499,
              "65f9c3542387dcad30bc52a9aac4557dc1c8f9d2157f8c95bd4ad9848cbe8a22"),
    )),
    ServerBuild("metal", "Apple Silicon (Metal)", "mac", (
        Asset("llama-b11512-bin-macos-arm64.tar.gz", 12_047_364,
              "90867f2d9ac9d409ad08257e9a7a076b5c65f5ae21f1fea13fe146b772cea66f"),
    )),
    ServerBuild("linux-cpu", "Processor only (Linux)", "linux", (
        Asset("llama-b11512-bin-ubuntu-x64.tar.gz", 17_793_407,
              "cf4083d1e89ccce41157b096d5c9d36e2ef4c8751cf96b709ed533a3a4fe98d3"),
    )),
    ServerBuild("linux-vulkan", "Any graphics card (Linux, Vulkan)", "linux", (
        Asset("llama-b11512-bin-ubuntu-vulkan-x64.tar.gz", 31_745_465,
              "ed1fb59f61741746b4a8fafcda3dd027e5af4ec585fc0492a1fa76660dbae7e0"),
    )),
)}


@dataclass(frozen=True)
class ModelChoice:
    key: str
    label: str
    repo: str
    revision: str  # pinned Hugging Face commit
    file: str
    size: int
    sha256: str
    vram_gb: float  # graphics memory it needs to run fully on the card (weights + working memory for 2 stories at once)
    note: str = ""
    family: str = "qwen"  # how "thinking" is switched off

    @property
    def size_gb(self) -> float:
        return round(self.size / 1e9, 1)


# Smallest first. "auto" picks the biggest one that fits the card's free memory (after keeping room for TV transcription).
MODELS: tuple[ModelChoice, ...] = (
    ModelChoice("qwen3.5-0.8b", "Qwen3.5 0.8B (tiny, for testing)", "unsloth/Qwen3.5-0.8B-GGUF",
                "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-Q4_K_M.gguf", 532_517_120,
                "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517", 1.6,
                "Too small to judge news well - only for checking that Pro AI runs."),
    ModelChoice("qwen3.5-4b", "Qwen3.5 4B", "unsloth/Qwen3.5-4B-GGUF",
                "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q4_K_M.gguf", 2_740_937_888,
                "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4", 4.2,
                "For 8 GB cards."),
    ModelChoice("qwen3.5-9b", "Qwen3.5 9B", "unsloth/Qwen3.5-9B-GGUF",
                "3885219b6810b007914f3a7950a8d1b469d598a5", "Qwen3.5-9B-Q4_K_M.gguf", 5_680_522_464,
                "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8", 7.4,
                "For 12-16 GB cards (e.g. RTX 5070 Ti) with TV transcription on."),
    ModelChoice("qwen3.5-9b-q6", "Qwen3.5 9B (higher quality)", "unsloth/Qwen3.5-9B-GGUF",
                "3885219b6810b007914f3a7950a8d1b469d598a5", "Qwen3.5-9B-Q6_K.gguf", 7_458_301_152,
                "91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59", 9.2,
                "For 16 GB cards when TV transcription is off."),
    ModelChoice("gemma-4-12b", "Gemma 4 12B", "google/gemma-4-12B-it-qat-q4_0-gguf",
                "29d097773436b69ff9feafd636ab4cf873786537", "gemma-4-12b-it-qat-q4_0.gguf", 6_975_879_296,
                "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b", 9.4,
                "Google's model; an alternative for 16 GB cards.", family="gemma"),
    ModelChoice("gemma-4-26b-a4b", "Gemma 4 26B-A4B", "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
                "7b92b5b28818151e8669af2e45e88d6086f490dd", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                14_249_047_104, "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891", 16.8,
                "For 24 GB cards (e.g. RTX 3090 / 4090). Fast: only part of it works on each word.", family="gemma"),
    ModelChoice("qwen3.8-27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF",
                "4ca720788d1e01f1bff70c033e0d0028fd02e502", "Qwen3.8-27B-UD-Q4_K_M.gguf", 16_464_440_224,
                "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482", 19.0,
                "Highest quality for 24 GB cards, about twice as slow."),
    ModelChoice("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF",
                "a483e9e6cbd595906af30beda3187c2663a1118c", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", 22_134_528_992,
                "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61", 24.8,
                "For 32 GB cards (e.g. RTX 5090) and Macs with 48 GB+. Fast."),
)
MODELS_BY_KEY = {m.key: m for m in MODELS}
# never picked automatically
NOT_AUTO = {"qwen3.5-0.8b", "qwen3.8-27b", "gemma-4-12b"}


def hf_url(model: ModelChoice) -> str:
    return f"https://huggingface.co/{model.repo}/resolve/{model.revision}/{model.file}"
