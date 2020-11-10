from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl

class WandbLoggerFrozenVal(pl.loggers.WandbLogger):
    @pl.utilities.rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        # nthomas - this is the key change allow_val_change=False
        self.experiment.config.update(params, allow_val_change=False)
