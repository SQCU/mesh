import json
import threading
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer
from solver.strat.joracle.server import Viewer, Handler
report=json.loads(Path('.build/continuation-validation/runtime-verified.json').read_text())
viewer=Viewer(Path(report['directory'])/'telemetry.jsonl',30)
viewer.refresh()
server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
server.viewer=viewer
thread=threading.Thread(target=server.serve_forever,daemon=True)
thread.start()
results={}
try:
    for route in ('/api/j','/api/policy','/j','/policy'):
        with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}{route}') as response:
            body=response.read()
            entry={'status':response.status,'bytes':len(body)}
            if route.startswith('/api/'):
                data=json.loads(body)
                state=data['learning']['terminal_win']
                assert state['updates']>0 and state['completed_episodes']==report['completed_episodes'] and state['outcome']==0
                assert state['execution'].get('unmatched_execution_rows',0)==0
                entry.update(updates=state['updates'],completed_episodes=state['completed_episodes'],outcome=state['outcome'],producer_state=data['status']['producer_state'])
                if route=='/api/j':
                    entry['j_mass']=sum(row['mass'] for row in data['measure']['strata'])
                    entry['application_joins']=data['measure']['joins']
                    assert entry['j_mass']>0
            else:
                assert b'id="continuation"' in body
            results[route]=entry
finally:
    server.shutdown()
    server.server_close()
    thread.join()
Path('.build/continuation-validation/viewer-http.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
