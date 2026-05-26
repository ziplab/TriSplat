from .decoder import Decoder
from .decoder_triangle_splatting_cuda import DecoderTriangleSplattingCUDA, DecoderTriangleSplattingCUDACfg

DECODERS = {
    "triangle_splatting_cuda": DecoderTriangleSplattingCUDA,
}

DecoderCfg = DecoderTriangleSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
