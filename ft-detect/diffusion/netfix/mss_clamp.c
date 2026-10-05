// LD_PRELOAD shim: cap the TCP segment size of every IPv4/IPv6 stream socket this process creates.
// For hosts whose interface MTU (9200) is larger than what the path between them passes (1500):
// without root the MTU cannot be changed, but TCP_MAXSEG set before connect/listen is unprivileged
// and is also announced to the peer, so both directions use small segments.
// Build: gcc -shared -fPIC -O2 -o mss_clamp.so mss_clamp.c -ldl
// Use:   LD_PRELOAD=/path/mss_clamp.so MSS_CLAMP=1400 <command>
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <sys/socket.h>

static int (*real_socket)(int, int, int);

int socket(int domain, int type, int protocol) {
    if (!real_socket) real_socket = (int (*)(int, int, int))dlsym(RTLD_NEXT, "socket");
    int fd = real_socket(domain, type, protocol);
    int base = type & ~(SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd >= 0 && (domain == AF_INET || domain == AF_INET6) && base == SOCK_STREAM) {
        const char *e = getenv("MSS_CLAMP");
        int mss = e ? atoi(e) : 1400;
        setsockopt(fd, IPPROTO_TCP, TCP_MAXSEG, &mss, sizeof(mss));
    }
    return fd;
}
