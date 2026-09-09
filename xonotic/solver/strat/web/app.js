'use strict';
const $=id=>document.getElementById(id), page=document.querySelector('main').id;
let arms=[];
const colors=['#52dcc3','#ffbd72'];
const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=x=>typeof x==='number'&&Number.isFinite(x)?Number(x.toPrecision(5)).toString():'—';
const ms=x=>num(typeof x==='number'?x*1000:NaN);
let data=null, timer=null, lastSequence=null, polling=false;
function table(id,heads,rows){$(id).innerHTML='<thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+esc(typeof v==='number'?num(v):v)+'</td>').join('')+'</tr>').join('')+'</tbody>';}
function canvas(id){const c=$(id),w=Math.max(240,c.clientWidth),h=210,d=Math.min(devicePixelRatio||1,2);c.width=w*d;c.height=h*d;const g=c.getContext('2d');g.scale(d,d);g.font='11px monospace';return {g,w,h};}
function plot(id,series){
 const {g,w,h}=canvas(id), valid=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
 if(!valid.length){g.fillStyle='#91a7bb';g.fillText('No measured points in this scope',12,30);return;}
 const [x0,x1,y0,y1]=valid.reduce(([a,b,c,d],[x,y])=>[Math.min(a,x),Math.max(b,x),Math.min(c,y),Math.max(d,y)],[Infinity,-Infinity,0,0]);
 const X=x=>48+(w-65)*(x-x0)/(x1-x0||1),Y=y=>h-30-(h-60)*(y-y0)/(y1-y0||1);
 g.strokeStyle='#283a4b';g.fillStyle='#91a7bb';
 for(let i=0;i<=4;i++){let y=y0+(y1-y0)*i/4;g.beginPath();g.moveTo(46,Y(y));g.lineTo(w-12,Y(y));g.stroke();g.fillText(num(y),2,Y(y)-3);}
 series.forEach((s,k)=>{g.strokeStyle=s.color||colors[k%2];g.fillStyle=g.strokeStyle;if(50+(k+1)*155<w)g.fillText(s.name,50+k*155,14);g.beginPath();let joined=false;s.points.forEach(([x,y])=>{if(!Number.isFinite(y)){joined=false;return;}if(joined)g.lineTo(X(x),Y(y));else g.moveTo(X(x),Y(y));joined=true;});g.stroke();s.points.filter(p=>Number.isFinite(p[1])).forEach(([x,y])=>g.fillRect(X(x)-1,Y(y)-1,3,3));});
 g.fillStyle='#91a7bb';g.fillText(num(x0),48,h-8);g.fillText(num(x1),w-65,h-8);
}
function heat(id,rows,signed=false){const {g,w,h}=canvas(id);if(!rows?.length){g.fillStyle='#91a7bb';g.fillText('No measured matrix',12,30);return;}const n=rows.length,m=Math.max(...rows.map(r=>r.length)),max=rows.reduce((a,r)=>r.reduce((a,v)=>Number.isFinite(v)?Math.max(a,Math.abs(v)):a,a),1);rows.forEach((r,i)=>r.forEach((v,j)=>{g.fillStyle=Number.isFinite(v)?'hsl('+(signed&&v<0?32:166)+' 65% '+(10+55*Math.abs(v)/max)+'%)':'#612e39';g.fillRect(j*w/m,i*h/n,Math.ceil(w/m),Math.ceil(h/n));}));}
function options(id,items){const el=$(id),selected=el.value,html=items.map(([key,label])=>'<option value="'+esc(key)+'">'+esc(label)+'</option>').join('');if(el.innerHTML!==html){el.innerHTML=html;if(items.some(x=>String(x[0])===selected))el.value=selected;}}
function renderContinuation(){
 table('continuation',['Policy','Updates','Episode','First applications','Unmatched applications','Pending returns','Awaiting outcomes','Outcome','VM faults','History bytes','Spilled requests','Journal bytes','Coalesced snapshots','Dropped frames'],arms.map(a=>{const l=data.learning?.[a]||{},e=l.execution||{},t=l.transport||{};return [a,l.updates,l.episode_id,e.first_execution_rows,e.unmatched_execution_rows,l.pending_states,l.awaiting_outcomes,l.outcome,l.view_faults,e.bytes,e.spilled_requests,l.journal_bytes,l.snapshots_superseded,t.dropped_frames];}));
}
function renderJ(){
 const m=data.measure||{},strata=m.strata||[];
 options('stratum',strata.map(s=>[s.key,(s.j_labels?.[0]||'unknown policy')+' · '+s.feature_width+'×'+s.j_width+' · n='+s.mass]));
 const s=strata.find(s=>s.key===(m.selected_key||$('stratum').value));
 $('stratum-info').textContent=s?'retained '+s.mass+' rows; finite '+s.finite_mass+'; J width '+s.j_width+'; measure age '+num(Date.now()/1000-m.sampled_at)+'s · coordinates '+num(s.coordinate_offset)+'–'+num(s.coordinate_offset+(s.coordinates?.length||0))+' of '+num(s.coordinate_matches)+' matching features':'No producer J stratum yet';
 plot('variance',[{name:'J variance',points:(s?.j_variance||[]).map((v,i)=>[(s.j_offset||0)+i,v])}]);
 $('rank').textContent='Native J width '+num(s?.j_width)+' · numerical rank '+num(s?.rank)+' · finite rows '+num(s?.finite_mass);
 plot('spectrum',[{name:'Singular value',points:(s?.singular_values||[]).map((v,i)=>[i,v])}]);
 const sampled=data.model?.row_outputs||[],keys=[...new Set(sampled.map(r=>r.arm+':'+(r.j_width??r.j.length)))];
 options('sample-arm',keys.map(k=>[k,k]));const rows=sampled.filter(r=>r.arm+':'+(r.j_width??r.j.length)===$('sample-arm').value);
 heat('selected-j',rows.map(r=>r.j),true);
 $('sample-info').textContent='Sampled response '+num(data.model?.response)+' · responder elapsed '+num(data.model?.t)+' · '+(data.model?.row_identity||'participant row index')+' '+rows.map(r=>r.row).join(', ')+' · coordinate offset '+num(data.model?.coordinate_window?.offset)+' · exact slice';
 const query=$('feature-filter').value.toLowerCase();
 table('features',['Exact feature coordinate','Variance','‖cov(feature,J)‖₂','Affine residual²'],(s?.coordinates||[]).filter(r=>r.name.toLowerCase().includes(query)).map(r=>[r.name,r.variance,r.covariance_norm,r.residual]));
 options('outcome-arm',[['all','All labelled policies'],...arms.map(a=>[a,a])]);
 table('outcomes',['Policy','Outcome / channel','Mass','Integral','Mean','Variance','‖cov(J,outcome)‖₂'],(m.outcomes||[]).filter(r=>$('outcome-arm').value==='all'||r.arm===$('outcome-arm').value).map(r=>[r.arm,r.name,r.mass,r.integral,r.mean,r.variance,r.covariance_norm]));
 const j=m.joins||{};$('joins').textContent=['state_application','applied','event'].map(k=>k+' joined '+num(j[k+'_joined_mass'])+' / '+num(j[k+'_mass'])).join(' · ');
 const last=data.series?.at(-1);options('cart-choice',[['all','All '+(last?.carts?.length||0)+' carts'],...(last?.carts||[]).map(c=>[c.id,'Cart '+c.id])]);plot('carts',(last?.carts||[]).filter(c=>$('cart-choice').value==='all'||String(c.id)===$('cart-choice').value).map((c,i)=>({name:'cart '+c.id,color:'hsl('+(i*137.5%360)+' 65% 65%)',points:data.series.map(s=>[s.t,s.carts.find(x=>x.id===c.id)?.depth])})));heat('focus',data.focus);
 table('assignments',['Player','Team','Controller','Behavior','Policy','State width','Rate norm','Residual norm','Requested seq','Applied seq'],(data.assignments||[]).map(r=>[r.edict,r.team,r.controller,r.behavior,r.policy_arm,r.state_width,r.velocity_l2,r.residual_l2,r.request_seq,r.applied_response_seq]));
}
function renderPolicy(){
 const series=data.series||[], first=series[0], last=series.at(-1), span=last?.t-first?.t, work=data.work||{};
 $('compute').textContent='Response compute '+ms(work.elapsed_s)+' ms / '+ms(work.deadline_s)+' ms deadline · optimizer '+ms(work.optimization?.elapsed_s)+' ms · '+num(work.optimization?.gradient_steps)+' gradient steps in latest group';
 $('policies').innerHTML=arms.map(arm=>{const l=data.learning?.[arm]||{},u=data.last_updates?.[arm]||{};return '<div class="card"><h2>'+esc(arm)+'</h2><div class="number">'+num(l.updates)+'</div><p>lifetime optimizer updates · '+num(span>0?(last.learning?.[arm]?.updates-first.learning?.[arm]?.updates)/span:null)+' updates / responder second<br>'+num(l.completed_episodes)+' completed episodes · '+num(l.truncated_episodes)+' truncated episodes</p><p>Last loss '+num(u.loss)+' · actor '+num(u.loss_pg)+' · entropy '+num(u.state_entropy)+'</p><p>'+num(u.schedule_updates)+' updates under '+esc(u.training_contract?.update_schedule)+'<br>last batch: '+num(u.batch)+' fresh / '+num(u.replay_batch)+' historical · '+num(u.gradient_steps)+' step<br>'+num(u.actor_rows)+' actor rows / '+num(u.value_rows)+' value rows<br>gradient norm '+num(u.gradient_norm)+' · response '+num(u.response)+'<br><small>'+esc(u.directory)+'</small></p></div>';}).join('');
 const comparison=data.comparison||{}, sample=data.comparison_vector_sample||{};
 $('comparison-scope').textContent='Same observed input · response '+num(comparison.response)+' · age '+num(comparison.sampled_at?Date.now()/1000-comparison.sampled_at:null)+'s · computation '+ms(comparison.elapsed_s)+' ms · full-vector sample response '+num(sample.response);
 table('divergence',['Player','Team','Assigned controller','Policy pair','KL left → right','KL right → left','Mean-rate distance','Sampled-rate distance','Residual distance'],(comparison.pairs||[]).map(r=>[r.player,r.team,r.assigned_policy,r.left+' / '+r.right,r.rate_kl_relation==='mutually_singular'?'∞':r.rate_kl_left_right_nats,r.rate_kl_relation==='mutually_singular'?'∞':r.rate_kl_right_left_nats,r.rate_mean_l2,r.sampled_rate_l2,r.residual_l2]));
 table('counterfactual',['Player','Team','Assigned policy','Evaluated policy','Version','State width','Rate mean norm','Rate scale RMS','Winner value','Loser value'],(comparison.rows||[]).filter(r=>$('comparison-rows').value==='all'||!r.controls_this_player).map(r=>[r.player,r.team,r.assigned_policy,r.evaluated_policy,r.policy_updates,r.state_width,r.rate_mean_l2,r.rate_sigma_rms,r.winner_value,r.loser_value]));
 table('reporting',['Report','State','Observations','Source / missing dependency'],(data.status.reports||[]).map(r=>[r.report,r.state,r.observations,r.source]));
 const study=data.study||{},ratings=data.composition_ratings||{},groups=ratings.groups||[];
 $('rating-scope').textContent=num(ratings.completed_rounds)+' completed rounds · '+num(ratings.configurations)+' realized bot configurations · '+num(ratings.exposure_records)+' exposure intervals · '+(ratings.excluded_rounds||[]).length+' rounds with incomplete attribution. '+(ratings.scope||'Awaiting attributed outcomes.')+' Paired-study rounds: '+num(study.paired_round_count)+'.';
 table('composition-ratings',['Map','Composition','Elo','Posterior SD','Rounds','Identified rank / parameters','Converged'],groups.flatMap(g=>(g.team_ratings||[]).map(r=>[g.context?.map,r.composition,r.elo,r.posterior_sd_elo,g.rounds,g.rank+' / '+g.parameters,g.converged])));
 table('factor-ratings',['Map','Bot configuration / controller','Elo offset','Posterior SD','Identified fraction'],groups.flatMap(g=>(g.ratings||[]).map(r=>[g.context?.map,r.factor,r.elo_offset,r.posterior_sd_elo,r.identified_fraction])));

 const metric=$('metric').value;plot('loss',arms.map(name=>({name,points:(data.series||[]).map(r=>[r.t,r.policy_updates?.[name]?.gradient_steps?r.policy_updates[name][metric]:null])})));
 plot('updates',arms.map(name=>({name,points:series.map(r=>[r.t,r.learning?.[name]?.updates])})));
 table('optimizer',['Policy','Total loss','Actor loss','Winner value','Loser value','Rate entropy','Gradient norm','Clip norm'],arms.map(a=>{const u=data.last_updates?.[a]||{};return [a,u.loss,u.loss_pg,u.loss_w,u.loss_l,u.state_entropy,u.gradient_norm,u.gradient_clip];}));
 table('rollout',['Policy','Behavior age (updates)','Actor rows','Value rows','Importance mean','Clipped fraction','Effective sample fraction'],arms.map(a=>{const u=data.last_updates?.[a]||{},r=u.importance_ratio||{};return [a,u.behavior_age_updates,u.actor_rows,u.value_rows,r.mean,r.clipped_fraction,r.effective_sample_fraction];}));
 table('replay',['Policy','Retained states','Configurations','Match groups','Evictions','Pending fresh states','History age (updates)','Target variance'],arms.map(a=>{let l=data.learning?.[a]||{},u=data.last_updates?.[a]||{};return [a,l.replay_size,l.configurations,l.matches,l.evictions,l.pending_states,u.replay_behavior_age_updates,u.value_target_variance];}));

 $('round-scope').textContent='Coverage: '+data.round_coverage+'; '+data.rounds.length+' rounds observed; current human rows '+num(data.human_rows)+'.';
 table('rounds',['Launch directory','Engine time','Winning team','Assigned winner arm','Strategy / terminal team exposure'],data.rounds.slice(-30).map(r=>{let t=r.event.actor_team,a=r.team_policy_arms||[];return [r.directory,r.event.time,t,t>0?a[t-1]:'draw',arms.map(arm=>a.filter(x=>x===arm).length).join(' / ')];}));
 $('history-scope').textContent=data.history.length+' completed launches retained across observed runs. Ratings pool attributed outcomes without recency weighting.';
 table('history',['Launch','Map','Teams','Carts','Strategy updates','Terminal updates'],data.history.slice(-50).map(r=>[r.ordinal,r.map,r.teams,r.carts,r.learning?.matrix_fusion?.updates,r.learning?.terminal_win?.updates]));
}
function render(){if(!data?.status?.sequence)return;arms=Object.keys(data.learning||{});renderContinuation();const s=data.status;$('source').textContent='Source: '+(s.host||'unknown host')+' · '+(s.run_directory?.split('/').at(-1)||s.directory||'no run selected');$('source').title=s.run_directory||s.directory||'';$('game').textContent=s.environment+' · '+s.teams+' teams · '+s.carts+' carts · '+s.players+' players · response '+s.response;page==='j'?renderJ():renderPolicy();}
async function poll(){
 clearTimeout(timer);if(polling)return;polling=true;
 try{
  const statusResponse=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(8000)});
  if(!statusResponse.ok)throw Error('HTTP '+statusResponse.status);
  const s=await statusResponse.json(),revision=s.viewer_id+':'+s.content_revision;
  if(!data||revision!==lastSequence){
   const query=page==='j'?'?'+new URLSearchParams({stratum:$('stratum').value,filter:$('feature-filter').value,offset:$('coordinate-offset').value,width:$('coordinate-width').value}):'';
   const response=await fetch('/api/'+page+query,{cache:'no-store',signal:AbortSignal.timeout(8000)});
   if(!response.ok)throw Error('HTTP '+response.status);
   data=await response.json();lastSequence=data.status.viewer_id+':'+data.status.content_revision;render();
  }
  const state=Date.now()/1000-s.sampled_at>15?'stale':s.producer_state;
  document.body.dataset.producer=state||'unknown';
  $('notice').textContent=(state==='stale'?'Telemetry has stopped advancing. Showing the last measured values. ':state==='awaiting_first_frame'?'No telemetry frames have arrived from this run yet. ':'')+(s.pending_directory?'Waiting for the first complete frame from '+s.pending_directory.split('/').at(-1)+'.':'');
  $('status').textContent=(state||'unknown')+' · sample '+num(s.sequence)+' · source file age '+num(s.frame_written_at?Date.now()/1000-s.frame_written_at:null)+'s'+(s.last_error?' · '+s.last_error:'')+Object.entries(s.report_errors||{}).map(([name,error])=>' · '+name+': '+error).join('');
 }catch(e){document.body.dataset.producer='stale';$('status').textContent='Read failed: '+e.message;}
 finally{polling=false;timer=setTimeout(poll,document.hidden?10000:2000);}
}

document.querySelectorAll('select,input').forEach(el=>el.addEventListener('input',()=>{if(page==='j'&&['stratum','feature-filter','coordinate-offset','coordinate-width'].includes(el.id)){lastSequence=null;poll();}else render();}));
document.addEventListener('visibilitychange',poll);window.addEventListener('resize',render);poll();
