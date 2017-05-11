"""Trains an Miq model.

Usage:
  Start training:
    python quality/miq_train.py --data_globs "/focus0/*,/focus1/*,/focus2/*, \
      /focus3/*,/focus4/*,/focus5/*,/focus6/*,/focus7/*,/focus8/*,/focus9/*, \
      /focus10/*" --train_log_dir <path_to_train_directory>

  View training progress:
    tensorboard --logdir=<path_to_train_directory>

    In web browser, go to localhost:6006.
"""
