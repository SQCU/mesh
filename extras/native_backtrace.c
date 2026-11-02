// extras/native_backtrace.c: the faulting thread's native stack on SIGABRT, SIGSEGV and SIGBUS, written to stderr before
// the signal's default action. A failed assertion's abort() runs this on the asserting thread itself, so the record names
// the code that asserted (2026-11-01: a rank's "commit an already committed command buffer" left no crash report and a
// Python dump of an unrelated thread). Loaded into a program with DYLD_INSERT_LIBRARIES; Python's faulthandler, set up
// after it, restores this handler and raises again once it has written its own dump.
//
//   cc -shared -O1 -o native_backtrace.dylib extras/native_backtrace.c
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void dump(int sig) {
    void *frames[96];
    char head[160];
    uint64_t thread = 0;
    pthread_threadid_np(NULL, &thread);
    const int n = backtrace(frames, 96);
    const int length = snprintf(head, sizeof head, "NATIVE-BACKTRACE signal %d thread %llu pid %d frames %d\n", sig,
                                (unsigned long long)thread, (int)getpid(), n);
    write(2, head, (size_t)length);
    backtrace_symbols_fd(frames, n, 2);
    write(2, "NATIVE-BACKTRACE end\n", 21);
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void install(void) {
    signal(SIGABRT, dump);
    signal(SIGSEGV, dump);
    signal(SIGBUS, dump);
}
