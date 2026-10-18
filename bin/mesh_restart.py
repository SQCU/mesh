"""A resident service's restart on a change of what it was started from (README.md, launchd): its process exits once any
of `paths` is written, replaced or removed (a kqueue vnode event), and launchd's KeepAlive starts it again on the new
files, so its definition never changes."""
import os, select, threading


def on_change(*paths):
    watch, events = select.kqueue(), []
    for path in dict.fromkeys(os.path.realpath(path) for path in paths if path):
        try: fd = os.open(path, getattr(os, "O_EVTONLY", os.O_RDONLY))
        except OSError: continue
        events.append(select.kevent(fd, select.KQ_FILTER_VNODE, select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                    select.KQ_NOTE_WRITE | select.KQ_NOTE_DELETE | select.KQ_NOTE_RENAME))
    def changed():
        watch.control(events, 1)
        os._exit(0)
    threading.Thread(target=changed, daemon=True).start()
