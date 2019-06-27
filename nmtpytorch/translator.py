# -*- coding: utf-8 -*-
import sys
import math
import time
import logging
from pathlib import Path

import torch

from .utils.misc import load_pt_file
from .utils.filterchain import FilterChain
from .utils.data import make_dataloader
from .utils.topology import Topology
from .utils.device import DEVICE

from . import models
from .config import Options
from .search import beam_search

logger = logging.getLogger('nmtpytorch')


class Translator:
    """A utility class to pack translation related features."""

    def __init__(self, **kwargs):
        # Store attributes directly. See bin/nmtpy for their list.
        self.__dict__.update(kwargs)

        for key, value in kwargs.items():
            logger.info('-- {} -> {}'.format(key, value))

        # How many models?
        self.n_models = len(self.models)

        # Store each model instance
        self.instances = []

        # Disable gradient tracking
        torch.set_grad_enabled(False)

        # Create model instances and move them to device
        for model_file in self.models:
            data = load_pt_file(model_file)
            weights, _, opts = data['model'], data['history'], data['opts']
            opts = Options.from_dict(opts, override_list=self.override)

            # Create model instance
            instance = getattr(models, opts.train['model_type'])(opts=opts)

            if not instance.supports_beam_search:
                logger.error(
                    "Model does not support beam search. Try 'nmtpy test'")
                sys.exit(1)

            # Setup layers
            instance.setup(is_train=False)
            # Load weights
            instance.load_state_dict(weights, strict=False)
            # Move to device
            instance.to(DEVICE)
            # Switch to eval mode
            instance.train(False)
            self.instances.append(instance)

        # Split the string
        self.splits = self.splits.split(',')

        # Do some sanity-check
        self.sanity_check()

        # Setup post-processing filters
        eval_filters = self.instances[0].opts.train['eval_filters']

        if self.disable_filters or not eval_filters:
            logger.info('Post-processing filters disabled.')
            self.filter = lambda s: s
        else:
            logger.info('Post-processing filters enabled.')
            self.filter = FilterChain(eval_filters)

        # Can be a comma separated list of hardcoded test splits
        logger.info('Will translate "{}"'.format(self.splits))
        if self.source:
            # We have to have single split name in this case
            split_set = '{}_set'.format(self.splits[0])
            input_dict = self.instances[0].opts.data.get(split_set, {})
            logger.info('Input configuration:')
            for data_source in self.source.split(','):
                key, path = data_source.split(':', 1)
                input_dict[key] = Path(path)
                logger.info(' {}: {}'.format(key, input_dict[key]))
            # Overwrite config's set name
            self.instances[0].opts.data[split_set] = input_dict

    def sanity_check(self):
        if self.source and len(self.splits) > 1:
            logger.info('You can only give one split name when -S is provided.')
            sys.exit(1)

        eval_filters = set([i.opts.train['eval_filters'] for i in self.instances])
        assert len(eval_filters) < 2, "eval_filters differ between instances."

        if len(self.instances) > 1:
            logger.info('Make sure you ensemble models with compatible vocabularies.')

        # check that all instances can perform the task
        if self.task_id is not None:
            task = Topology(self.task_id)
            incl = [task.is_included_in(i.topology) for i in self.instances]
            assert False not in incl, \
                'Not all models are compatible with task "{}"!'.format(task.direction)

    def translate(self, split):
        """Returns the hypotheses generated by translating the given split
        using the given model instance.

        Arguments:
            split(str): A test split defined in the .conf file before
                training.

        Returns:
            list:
                A list of optionally post-processed string hypotheses.
        """

        # Load data
        dataset = self.instances[0].load_data(split, self.batch_size, mode='beam')

        # NOTE: Data iteration needs to be unique for ensembling
        # otherwise it gets too complicated
        loader = make_dataloader(dataset)

        logger.info('Starting translation')
        start = time.time()
        hyps = beam_search(self.instances, loader, task_id=self.task_id,
                           beam_size=self.beam_size, max_len=self.max_len,
                           lp_alpha=self.lp_alpha, suppress_unk=self.suppress_unk,
                           n_best=self.n_best, wait_k=self.wait_k)
        up_time = time.time() - start
        logger.info('Took {:.3f} seconds, {} sent/sec'.format(
            up_time, math.floor(len(hyps) / up_time)))

        return hyps

    def dump(self, hyps, split):
        """Writes the results into output.

        Arguments:
            hyps(list): A list of hypotheses.
        """
        suffix = ""
        if self.lp_alpha > 0.:
            suffix += ".lp_{:.1f}".format(self.lp_alpha)
        if self.suppress_unk:
            suffix += ".no_unk"
        suffix += ".beam{}".format(self.beam_size)
        if self.n_best:
            suffix += ".nbest"
        output = "{}.{}{}".format(self.output, split, suffix)

        f = open(output, 'w')
        if self.n_best:
            for idx, (cands, scores) in enumerate(hyps):
                cands = self.filter(cands)
                sorted_cs = sorted(
                    zip(cands, scores), key=lambda x: x[1], reverse=True)
                for cand, score in sorted_cs:
                    # cands is a list of n sents, scores as well
                    f.write('{} ||| {} ||| {:.5f}\n'.format(idx, cand, score))
        else:
            # Post-process strings if requested
            hyps = self.filter(hyps)
            for line in hyps:
                f.write(line + '\n')
        f.close()

    def __call__(self):
        """Dumps the hypotheses for each of the requested split/file."""
        for input_ in self.splits:
            hyps = self.translate(input_)
            self.dump(hyps, input_)
