# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import logging
logger = logging.getLogger(__name__)

class EpochLogger(object):
    def __init__(self, save_file, precision=2):
        self.save_file = save_file
        self.precision = precision

    def item_to_string(self, key, value, prefix=None):
        if isinstance(value, float) and 1.0 < value < 100.0:
            value = f"{value:.{self.precision}f}"
        elif isinstance(value, float):
            value = f"{value:.{self.precision}e}"
        if prefix is not None:
            key = f"{prefix} {key}"
        return f"{key}: {value}"

    def stats_to_string(self, stats, prefix=None):
        return ", ".join(
            [self.item_to_string(k, v, prefix) for k, v in stats.items()]
        )

    def log_stats(
        self,
        stats_meta,
        stats,
        stage='train',
        verbose=True,
    ):
        string = self.stats_to_string(stats_meta)
        if stats is not None:
            string += " - " + self.stats_to_string(stats, stage)

        with open(self.save_file, "a") as fw:
            print(string, file=fw)
        if verbose:
            logger.info(string)


class EpochCounter(object):
    def __init__(self, limit):
        self.current = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            logger.info(f"Going into epoch {self.current}")
            return self.current
        raise StopIteration

    def save(self, path, device=None):
        with open(path, "w") as f:
            f.write(str(self.current))

    def load(self, path, device=None):
        with open(path) as f:
            saved_value = int(f.read())
            self.current = saved_value
