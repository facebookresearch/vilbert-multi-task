#!/usr/bin/env python
import glob
import os
import os.path as pth


def main():
    '''
    Quick and dirty helper to manage tensorboard events.
    Commands:
        read - read all the files in an event dir
        clean - clean up older event files in directories with multiple fevents

    Usage:
        event_helper.py <command> [<event_dir>]
    '''
    import docopt, textwrap
    args = docopt.docopt(textwrap.dedent(main.__doc__))

    cmd = args['<command>']
    if args['<event_dir>'] is None:
        event_dirs = list(glob.glob('logs/*'))
    else:
        event_dirs = [args['<event_dir>']]
    # event_dirs = [ed for ed in event_dirs if pth.basename(ed).startswith('exp')]
    assert cmd in ['read', 'clean']

    if cmd == 'read':
        import tensorflow as tf
        event_fnames = []
        for event_dir in event_dirs:
            evts = glob.glob(pth.join(event_dir, '*'))
            event_fnames += [f for f in evts if 'tfevents' in f]
        for fname in event_fnames:
            print(fname)
            for summary in tf.train.summary_iterator(fname):
                print(summary)
    elif cmd == 'clean':
        for event_dir in event_dirs:
            event_fnames = glob.glob(pth.join(event_dir, '*'))
            event_fnames.sort(key=pth.getctime, reverse=True)
            #print('not removing {}'.format(event_fnames[:1]))
            for fname in event_fnames[1:]:
                print('removing {}'.format(fname))
                os.remove(fname)


if __name__ == '__main__':
    main()
