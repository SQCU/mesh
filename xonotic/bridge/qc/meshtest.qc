void(string s, ...) dprint2 = #339;
float(float tmr) gettime = #519;
string(float f) ftos2 = #26;
entity() spawn2 = #14;

float(string name, float nreq, float nresp) mesh_open = #644;
void(float h) mesh_close = #645;
void(float h, float i, float v) mesh_set = #646;
float(float h, float i) mesh_get = #647;
void(float h, float dst, .float fld, float firstedict, float numedicts) mesh_gather = #648;
void(float h, float src, .float fld, float firstedict, float numedicts) mesh_scatter = #649;
float(float h) mesh_publish = #650;
float(float h) mesh_poll = #651;
float(float h, float seq, float usec) mesh_wait = #652;
float(float h, float sel) mesh_stat = #653;

.float slot;
.float out;

float NB = 512;
float NF = 4096;
float NFRAMES = 600;

float h, i, k, t0, t1, seq, got, prev, frames, started;
float acc_bridge, acc_lag, nlag, bad, worst, lastseen;
entity e;

void() setup =
{
	h = mesh_open("/mesh_qc", NF, NF);
	dprint2("MESH handle ", ftos2(h), " nreq ", ftos2(mesh_stat(h, 3)), " nresp ", ftos2(mesh_stat(h, 4)), " depth ", ftos2(mesh_stat(h, 7)), "\n");
	for (i = 0; i < NB; i = i + 1)
	{
		e = spawn2();
		e.slot = i * 0.5 - 64;
	}
	e = spawn2();
	e.slot = 0;
	t0 = gettime(2);
	for (k = 0; k < 65536; k = k + 1)
		mesh_gather(h, 0, slot, 1, NB);
	t1 = gettime(2);
	dprint2("MESH gather_512_us ", ftos2((t1 - t0) * 1000000 / 65536), "\n");
	t0 = gettime(2);
	for (k = 0; k < 65536; k = k + 1)
		mesh_scatter(h, 0, out, 1, NB);
	t1 = gettime(2);
	dprint2("MESH scatter_512_us ", ftos2((t1 - t0) * 1000000 / 65536), "\n");
	t0 = gettime(2);
	for (k = 0; k < 1000000; k = k + 1)
		got = mesh_poll(h);
	t1 = gettime(2);
	dprint2("MESH poll_ns ", ftos2((t1 - t0) * 1000000000 / 1000000), "\n");
	t0 = gettime(2);
	for (k = 0; k < 65536; k = k + 1)
		seq = mesh_publish(h);
	t1 = gettime(2);
	dprint2("MESH publish_16KiB_us ", ftos2((t1 - t0) * 1000000 / 65536), "\n");
	t0 = gettime(2);
	for (k = 0; k < 65536; k = k + 1)
	{
		mesh_gather(h, 0, slot, 1, NB);
		mesh_set(h, NB, k);
		seq = mesh_publish(h);
		got = mesh_poll(h);
		mesh_scatter(h, 0, out, 1, NB);
	}
	t1 = gettime(2);
	dprint2("MESH full_cycle_us ", ftos2((t1 - t0) * 1000000 / 65536), "\n");
};

void() StartFrame =
{
	if (!started)
	{
		started = 1;
		setup();
		return;
	}
	if (frames >= NFRAMES)
		return;
	frames = frames + 1;

	t0 = gettime(2);
	mesh_gather(h, 0, slot, 1, NB);
	mesh_set(h, NB, frames);
	seq = mesh_publish(h);
	got = mesh_poll(h);
	if (got > lastseen)
		mesh_scatter(h, 0, out, 1, NB);
	t1 = gettime(2);
	acc_bridge = acc_bridge + (t1 - t0);

	if (got > lastseen)
	{
		prev = seq - got;
		lastseen = got;
		acc_lag = acc_lag + prev;
		nlag = nlag + 1;
		if (prev > worst)
			worst = prev;
		e = nextent(world);
		if (e.out != e.slot * 2 + 1)
			bad = bad + 1;
	}

	if (frames == NFRAMES)
	{
		dprint2("MESH frames ", ftos2(frames), " published ", ftos2(mesh_stat(h, 0)), " delivered ", ftos2(mesh_stat(h, 1)), "\n");
		dprint2("MESH responses ", ftos2(nlag), " mismatches ", ftos2(bad), " seqlock_retries ", ftos2(mesh_stat(h, 6)), " solver_served ", ftos2(mesh_stat(h, 5)), "\n");
		dprint2("MESH inframe_bridge_us_mean ", ftos2(acc_bridge * 1000000 / frames), "\n");
		dprint2("MESH delivery_lag_seq_mean ", ftos2(acc_lag / nlag), " worst ", ftos2(worst), "\n");
		dprint2("MESH DONE\n");
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
