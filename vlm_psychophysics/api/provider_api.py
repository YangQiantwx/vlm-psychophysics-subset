"""Known compatible endpoint routes."""

ROUTES = {
    "nvidia-internal": ("https://inference-api.nvidia.com/v1", "NVIDIA_API_KEY", None),
    "nvidia-public": (
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
        "moonshotai/kimi-k2.5",
    ),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "kimi-k3.1-pro"),
    "bigmodel": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "glm-4.6v"),
}
