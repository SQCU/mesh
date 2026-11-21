"""dataflow.svg/png: the routing policy's training data flow, one picture. python3 dataflow.py (qlmanage rasterizes)"""
import subprocess
from xml.sax.saxutils import escape

W, H = 1500, 1500   # square: qlmanage renders into a square
out = []
def box(x, y, w, h, title, lines, fill):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#333" stroke-width="1.5"/>')
    out.append(f'<text x="{x + 12}" y="{y + 24}" font-size="17" font-weight="bold">{escape(title)}</text>')
    for i, l in enumerate(lines):
        out.append(f'<text x="{x + 12}" y="{y + 48 + 19 * i}" font-size="14" xml:space="preserve">{escape(l)}</text>')
def arrow(x1, y1, x2, y2, label='', color='#333', dy=-6):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#a)"/>')
    if label:
        out.append(f'<text x="{(x1 + x2) / 2 + 6}" y="{(y1 + y2) / 2 + dy}" font-size="13" fill="{color}">{escape(label)}</text>')
def note(x, y, text, color):
    out.append(f'<text x="{x}" y="{y}" font-size="13" fill="{color}">{escape(text)}</text>')

SYN, ROLL, LOAD, TRAIN = '#e8f1fb', '#eef7e8', '#fdf3e3', '#f6e9f5'
out.append('<text x="20" y="34" font-size="22" font-weight="bold">The routing policy&apos;s training: the whole data flow</text>')
note(20, 58, 'black arrows: the data.  Below: every wait the code has now (green), and what it replaced (red, as of 8168f512).', '#555')

box(20, 80, 430, 200, '1  SYNTHESIS: cases (py/route/cases.py, numpy)', [
    'rng -> fabric (6 kinds) , stream (collectives,', '        arrivals on their floors), window', '        (dt, lag, refreshes, warm router, burst),',
    '        h0 (zero | sigma N(0,1) | pool), keys', 'out: case[n]  (struct route_case records)', 'pool  <-  states runs ended with'], SYN)

box(520, 80, 960, 380, '2  SYNTHESIS: rollouts (py/route/loop.py Rollouts.tick; sims.c, model.swift)', [
    'sim (sim.c, CPU, one per run)            policy forward (GPU, int8 minGRU)',
    '  fabric state at tau_k', '  -> tokens: lanes [L][IN], contexts [C][IN]   (route.h features)  ----->   h, tokens -> w [C][FS], h\'',
    '                                                                   <-----   w',
    '  pieces move; each draw (v, S): table over deg^|S| splits', '      logit = 10 tanh(w . phi / sqrt 40), masked, softmax, sample  (CPU, per draw)',
    '      phi = split features from the draw\'s live lane waits', '  run ends: costs (completion / floor), draws by refresh', '',
    'out (a run): h0, tokens_k, w_k, draws_k, costs     (kept in the simulators\' arena until loaded)',
    'a split\'s features, logit and table: split.h, ONE definition (the sampler, the credit, the checks\' score)',
    'two halves of the slots alternate: one simulates (CPU, one parallel pass) while the other\'s forward runs (GPU)'], ROLL)

box(20, 520, 430, 220, '3  LOADING: a learning batch (loop.load, numpy)', [
    'runs (4 a case)', '-> advantage: T - mean(T of the other 3)', '-> rows: each refresh\'s drawn contexts\' tokens',
    '-> segments (a refresh\'s rows), lane rows, h0', '-> credit inputs: draws, first draw a row,', '   behaviour w, task-0 scale', 'out: sims_export, one parallel copy into', 'the batch\'s shared buffers'], LOAD)

box(520, 520, 960, 220, '4  TRAINING (GPU, ONE command buffer a step: route_net_step)', [
    'forward again (the same network, theta now)  ->  w_cur', 'credit: per draw  ratio x (psi - E psi) under w_cur   (score function; ratio = pi_cur / pi_behaviour)',
    'backward through the refreshes (int8 products, three tasks: throughput, latency, queueing)', '-> grads [3][P],  gram [3][3]',
    'EXCHANGE (world 4): grads copied into the lease, summed across ranks (libnccl-mesh) in the same command buffer',
    'OPTIMIZER: gram -> UPGrad (upgrad.h, on the GPU) -> Adam -> pin V -> int8 codes  =  theta',
    'theta -> the actor: route_net_publish, two shared events (staged, consumed): no CPU wait'], TRAIN)

arrow(450, 180, 520, 180, 'case[n]')
arrow(980, 460, 980, 520, '')
arrow(760, 460, 450, 600, 'records')
arrow(450, 640, 520, 640, 'batch')
arrow(1300, 740, 1300, 800)
out.append('<path d="M1300 800 L1490 800 L1490 60 L1250 60 L1250 80" fill="none" stroke="#333" stroke-width="2.2" marker-end="url(#a)"/>')
note(1310, 790, 'theta -> the rollouts\' forward', '#333')
out.append('<path d="M640 460 L640 500 L235 500 L235 280" fill="none" stroke="#333" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#a)"/>')
note(250, 495, 'end states -> pool', '#333')

y = 790
note(20, y, 'The waits now (green), each the data\'s own; and what they replaced (red):', '#000')
rows = [
    ('per run, per refresh: 4 lock hand-offs, 2 broadcasts waking every worker, 55 KB copied into new arrays', 'none: one parallel pass a half (sims_step), tokens kept in the arena, packed once (sims_pack)'),
    ('per forward: a command buffer, completion by lock + broadcast; a service thread', 'route_net_wait on a half\'s forward, before that half simulates on (its w)'),
    ('per optimizer step: 3 CPU waits; theta copied to the producers WAITED behind their forwards', 'none: learn, exchange, optimizer one command buffer; theta by shared events'),
    ('per step: gram read back to the CPU for UPGrad; ratios summed on the CPU', 'none: UPGrad on the GPU; ratios summed in the step\'s completion handler'),
    ('per exchange: a thread, a CPU wait on the mesh, the gradient copied 3x; gram computed again', 'none on the CPU: the sum in the step\'s command buffer, through the lease'),
    ('per case: room/push handshake, caseLock x4, 12 KB struct copies into a queue', 'route_net_wait on a learning batch before it is filled again (two alternate)'),
]
for i, (bad, good) in enumerate(rows):
    note(20, y + 26 + 38 * i, '- ' + bad, '#b00020')
    note(40, y + 44 + 38 * i, '-> ' + good, '#1b7f2a')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">'
       '<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#333"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(out) + '</svg>')
open('dataflow.svg', 'w').write(svg)
subprocess.run(['qlmanage', '-t', '-s', str(W), '-o', '.', 'dataflow.svg'], capture_output=True)
subprocess.run(['mv', 'dataflow.svg.png', 'dataflow.png'])
