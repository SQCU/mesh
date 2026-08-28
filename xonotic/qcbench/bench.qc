void(string s, ...) dprint2 = #339;
float(float tmr) gettime = #519;
float() buf_create = #460;
string(float b, float i) bufstr_get = #466;
void(float b, float i, string s) bufstr_set = #467;
float(string s) stof2 = #81;
string(float f) ftos2 = #26;
float(string s) strlen2 = #114;
string(float b, string glue) buf_implode = #465;
entity() spawn2 = #14;

float(string name, float nfloats) mesh_open = #644;
void(float h) mesh_close = #645;
void(float h, float i, float v) mesh_set = #646;
float(float h, float i) mesh_get = #647;
void(float h, float dst, .float fld, float first, float n) mesh_gather = #648;
void(float h, float src, .float fld, float first, float n) mesh_scatter = #649;
float(float h) mesh_publish = #650;
float(float h) mesh_poll = #651;

.float slot;
float N, i, k, t0, t1, acc, b, h, seq, got, waits;
entity e;
void() bench = {
    N = 1000000;
    t0 = gettime(2); for (i = 0; i < N; i = i + 1) { acc = acc + i; } t1 = gettime(2);
    dprint2("emptyloop_ns_per_iter ", ftos2((t1 - t0) * 1000000000 / N), "\n");
    t0 = gettime(2); for (i = 0; i < N; i = i + 1) { acc = acc + strlen2("x"); } t1 = gettime(2);
    dprint2("builtin_strlen_ns_per_call ", ftos2((t1 - t0) * 1000000000 / N), "\n");

    b = buf_create();
    for (i = 0; i < 4096; i = i + 1) { bufstr_set(b, i, "1234.5678"); }
    t0 = gettime(2);
    for (k = 0; k < 64; k = k + 1) for (i = 0; i < 4096; i = i + 1) { acc = acc + stof2(bufstr_get(b, i)); }
    t1 = gettime(2);
    dprint2("A_bufstr_get_stof_per_4096_us ", ftos2((t1 - t0) * 1000000 / 64), "\n");
    t0 = gettime(2);
    for (k = 0; k < 64; k = k + 1) for (i = 0; i < 4096; i = i + 1) { bufstr_set(b, i, ftos2(acc)); }
    t1 = gettime(2);
    dprint2("B_ftos_bufstr_set_per_4096_us ", ftos2((t1 - t0) * 1000000 / 64), "\n");
    t0 = gettime(2);
    for (k = 0; k < 64; k = k + 1) { acc = acc + strlen2(buf_implode(b, " ")); }
    t1 = gettime(2);
    dprint2("C_buf_implode_4096_us ", ftos2((t1 - t0) * 1000000 / 64), "\n");

    h = mesh_open("/meshdemo", 8192);
    dprint2("mesh_handle ", ftos2(h), "\n");
    t0 = gettime(2);
    for (k = 0; k < 64; k = k + 1) for (i = 0; i < 4096; i = i + 1) { mesh_set(h, i, acc); }
    t1 = gettime(2);
    dprint2("D_mesh_set_per_4096_us ", ftos2((t1 - t0) * 1000000 / 64), "\n");
    t0 = gettime(2);
    for (k = 0; k < 64; k = k + 1) for (i = 0; i < 4096; i = i + 1) { acc = acc + mesh_get(h, 4096 + i); }
    t1 = gettime(2);
    dprint2("E_mesh_get_per_4096_us ", ftos2((t1 - t0) * 1000000 / 64), "\n");

    for (i = 0; i < 512; i = i + 1) { e = spawn2(); e.slot = i; }
    t0 = gettime(2);
    for (k = 0; k < 65536; k = k + 1) { mesh_gather(h, 0, slot, 1, 512); }
    t1 = gettime(2);
    dprint2("F_mesh_gather_512_us ", ftos2((t1 - t0) * 1000000 / 65536), "\n");
    t0 = gettime(2);
    for (k = 0; k < 65536; k = k + 1) { mesh_scatter(h, 4096, slot, 1, 512); }
    t1 = gettime(2);
    dprint2("G_mesh_scatter_512_us ", ftos2((t1 - t0) * 1000000 / 65536), "\n");
    t0 = gettime(2);
    for (k = 0; k < 1000000; k = k + 1) { acc = acc + mesh_poll(h); }
    t1 = gettime(2);
    dprint2("H_mesh_poll_ns ", ftos2((t1 - t0) * 1000000000 / 1000000), "\n");

    acc = 0;
    for (k = 0; k < 200; k = k + 1)
    {
        mesh_gather(h, 0, slot, 1, 512);
        t0 = gettime(2);
        seq = mesh_publish(h);
        waits = 0;
        while (mesh_poll(h) != seq && waits < 2000000) { waits = waits + 1; }
        t1 = gettime(2);
        mesh_scatter(h, 4096, slot, 1, 512);
        if (k > 20) acc = acc + (t1 - t0);
    }
    dprint2("J_inframe_solver_roundtrip_us ", ftos2(acc * 1000000 / 179), " lastspins ", ftos2(waits), "\n");
};
float did, frames, tstart;
void() StartFrame = {
    if (!did) { did = 1; bench(); seq = mesh_publish(h); tstart = gettime(2); frames = 0; }
    else if (seq)
    {
        frames = frames + 1;
        got = mesh_poll(h);
        if (got == seq)
        {
            dprint2("I_roundtrip_frames ", ftos2(frames), " wall_us ", ftos2((gettime(2) - tstart) * 1000000), " check ", ftos2(mesh_get(h, 4096)), "\n");
            if (waits < 20) { waits = waits + 1; seq = mesh_publish(h); tstart = gettime(2); frames = 0; }
            else seq = 0;
        }
    }
};
void() main = {};
void() PlayerPreThink = {};
void() PlayerPostThink = {};
void() ClientKill = {};
void() ClientConnect = {};
void() PutClientInServer = {};
void() ClientDisconnect = {};
void() SetNewParms = {};
void() SetChangeParms = {};
