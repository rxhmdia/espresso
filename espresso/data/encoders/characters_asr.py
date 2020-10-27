# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import List, Optional, Union

from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass
from fairseq.file_io import PathManager

from espresso.tools.utils import tokenize


@dataclass
class CharactersAsrConfig(FairseqDataclass):
    space_symbol: Optional[str] = field(
        default="<space>", metadata={"help": "space symbol"}
    )
    ends_with_space: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to append <space> to the end of each tokenized sentence"
        },
    )
    non_lang_syms: Optional[Union[str, List]] = field(
        default=None,
        metadata={
            "help": "List of non-linguistic symbols, or path to a file listing these "
            "symbols, e.g., <NOISE> etc. One entry per line. To be filtered out when "
            "calculating WER/CER."
        },
    )


@register_bpe("characters_asr", dataclass=CharactersAsrConfig)
class CharactersAsr(object):
    def __init__(self, cfg):
        self.space_symbol = cfg.space_symbol
        self.ends_with_space = cfg.ends_with_space
        if cfg.non_lang_syms is None:
            self.non_lang_syms = None
        elif isinstance(cfg.non_lang_syms, list):
            self.non_lang_syms = cfg.non_lang_syms
        else:
            try:
                with open(
                    PathManager.get_local_path(cfg.non_lang_syms), "r", encoding="utf-8"
                ) as fd:
                    self.non_lang_syms = [x.rstrip() for x in fd.readlines()]
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}".format(fd))

    def encode(self, x: str) -> str:
        y = tokenize(x, space=self.space_symbol, non_lang_syms=self.non_lang_syms)
        if self.ends_with_space:
            return y + " " + self.space_symbol
        else:
            return y

    def decode(self, x: str) -> str:
        return x.replace(" ", "").replace(self.space_symbol, " ").strip()
