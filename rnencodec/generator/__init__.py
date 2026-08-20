from .generator import RNNGenerator
from .generator import RNNGeneratorSoft

__all__ = ["RNNGenerator", "RNNGeneratorSoft", "EncodecRTPlayer"]


def __getattr__(name):
    """Lazily resolve EncodecRTPlayer (PEP 562).

    Importing it eagerly pulled `realtime_synth` — and through it PortAudio — into
    every `import rnencodec`, including training runs that never play audio. On a
    headless machine such as an HPC compute node, where PortAudio cannot be
    installed, that made `from training.loop import train_model` fail outright:

        rnencodec/__init__.py -> generator/__init__.py -> rnencodec_rtplayer
        -> realtime_synth -> ModuleNotFoundError

    Nothing in the training path references EncodecRTPlayer, so the dependency is
    deferred to first use. `from rnencodec.generator import EncodecRTPlayer` still
    works unchanged for real-time code (PEP 562 covers from-imports), and still
    raises the same ImportError when realtime_synth is genuinely missing — just at
    the point of use rather than at package import.
    """
    if name == "EncodecRTPlayer":
        from .rnencodec_rtplayer import EncodecRTPlayer
        return EncodecRTPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")