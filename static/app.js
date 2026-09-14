'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {mode:'text', templates:[], page:1, total:0, result:null, original:{work:'',answer:''}, urls:{}, busy:false};
let toastTimer, searchTimer;
function toast(message, error=false) {
  clearTimeout(toastTimer); $('toast').textContent=message; $('toast').className=error?'error':''; $('toast').hidden=false;
  toastTimer=setTimeout(()=>$('toast').hidden=true, error?8000:4500);
}
async function api(url, opts={}) {
  const controller = new AbortController(); const timeout=setTimeout(()=>controller.abort(), 180000);
  try {
    const response=await fetch(url,{...opts,signal:controller.signal});
    const data=await response.json();
    if(!response.ok) throw new Error(data.message||'请求失败，请稍后重试。');
    return data;
  } catch(e) { if(e.name==='AbortError') throw new Error('请求等待超时，请先查看批改记录，避免重复提交。'); throw e; }
  finally {clearTimeout(timeout);}
}
const jsonOpts=(method,value)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
function config() {return {engine:$('engine').value,maxScore:Number($('max-score').value),passPercent:Number($('pass-percent').value),useDeepseek:$('ai').checked,aiWeight:Number($('ai-weight').value)};}
function applyConfig(value) {
  $('engine').value=value.engine; $('max-score').value=value.maxScore; $('pass-percent').value=value.passPercent;
  $('ai').checked=state.mode!=='batch' && value.useDeepseek; $('ai-weight').value=value.aiWeight; aiVisibility();
}
function aiVisibility() { $('ai-options').hidden=!$('ai').checked; $('ai-weight-label').textContent=$('ai-weight').value+'%'; }
function rubricValue(id='rubric') {
  return $(id).value.split(/\r?\n/).filter(s=>s.trim()).map(line=>{
    const parts=line.split('|');
    if(parts.length>2 || (parts.length===2 && !parts[1].trim())) throw new Error('评分要点格式应为：关键词 | 权重。');
    return {keyword:parts[0].trim(),weight:parts.length===2?Number(parts[1].trim()):1};
  });
}
function rubricText(points=[]) {return points.map(p=>`${p.keyword} | ${p.weight}`).join('\n');}
function counts() {for(const field of ['work','answer']) $(field+'-count').textContent=$(field).value.length+' / 5000';}
function invalidateResult() {state.result=null; $('result-content').hidden=true; $('result-empty').hidden=false; $('result-state').textContent='等待批改';}
function baseline() {state.original={work:$('work').value,answer:$('answer').value}; counts(); invalidateResult();}
function setMode(mode) {
  if(state.busy) return;
  state.mode=mode;
  document.querySelectorAll('[data-mode]').forEach(b=>{const selected=b.dataset.mode===mode;b.classList.toggle('selected',selected);b.setAttribute('aria-selected',String(selected));});
  $('image-input').hidden=mode!=='image'; $('single-input').hidden=mode==='batch'; $('batch-input').hidden=mode!=='batch';
  $('student-field').hidden=mode==='batch'; $('student').required=mode!=='batch'; $('work').required=mode!=='batch';
  $('batch-text').required=mode==='batch'; $('ai').disabled=mode==='batch';
  if(mode==='batch') $('ai').checked=false;
  aiVisibility(); invalidateResult();
}
function lockWorkspace(locked) {
  state.busy=locked;
  document.querySelectorAll('#page-workspace input,#page-workspace textarea,#page-workspace select,#page-workspace button').forEach(el=>{
    if(locked){el.dataset.wasDisabled=String(el.disabled);el.disabled=true;}
    else {el.disabled=el.dataset.wasDisabled==='true';delete el.dataset.wasDisabled;}
  });
}
async function refreshStats() {
  const data=await api('/api/stats');
  $('stat-total').innerHTML=esc(data.total)+' <em>份</em>';
  $('stat-average').textContent=data.average===null?'—':data.average+'%';
  $('stat-pass').textContent=data.passRate===null?'—':data.passRate+'%';
  $('stat-templates').innerHTML=esc(data.templates)+' <em>个</em>';
  $('nav-count').textContent=data.total;
  const max=Math.max(...data.distribution,1);
  $('chart').innerHTML=data.distribution.map((n,i)=>`<div class="chart-col"><span>${n} 份</span><div class="chart-bar" style="height:${n/max*85}%"></div><small>${i*20}–${i===4?100:(i+1)*20-1}%</small></div>`).join('');
  $('chart-summary').textContent=data.total?`共 ${data.total} 份批改记录 · 平均得分率 ${data.average}% · 达标比例 ${data.passRate}%`:'还没有批改记录。完成第一份作业后，这里将显示真实统计。';
}
function resultMarkup(r, full=false) {
  const id=encodeURIComponent(r.id);
  let html=`<div class="result-name">${esc(r.studentName)} · ${esc(r.title)}</div><div class="score-display"><strong>${esc(r.score)}</strong><span>/ ${esc(r.options.maxScore)}</span></div><div class="result-tag ${r.passed?'':'review'}">${r.passed?'已达标':'待提升'} · 得分率 ${esc(r.similarity)}%</div><div class="score-breakdown"><span>基础评分 <b>${esc(r.basePercent)}%</b></span><span>AI 增强 <b>${r.aiPercent===null?'未使用':esc(r.aiPercent)+'%'}</b></span></div>`;
  html+=r.warnings.map(w=>`<div class="warning">${esc(w)}</div>`).join('');
  html+=`<p class="result-note">${esc(r.explanation)}</p>`;
  if(r.rubric.length) html+=`<div class="point-list">${r.rubric.map(p=>`<span class="point ${p.matched?'matched':''}">${p.matched?'✓':'○'} ${esc(p.keyword)}</span>`).join('')}</div>`;
  if(full) html+=`<h3>参考答案</h3><div class="full-text">${esc(r.answerContent)}</div><h3>学生作答</h3><div class="full-text">${esc(r.workContent)}</div>`;
  html+=`<details><summary class="text-button">查看文本差异</summary><div class="diff-legend">绿色为新增，红色为缺少的文字；颜色不代表答案正误。</div><div class="diff-box">${r.diff.map(d=>d.type==='equal'?esc(d.answer):`${d.reference?'<del>'+esc(d.reference)+'</del>':''}${d.answer?'<ins>'+esc(d.answer)+'</ins>':''}`).join('')}</div></details>`;
  html+=`<div class="result-actions"><a class="button secondary" href="/api/records/${id}/export?format=csv">导出 CSV</a><a class="button secondary" href="/api/records/${id}/export?format=json">JSON</a><a class="button secondary" target="_blank" rel="noopener" href="/records/${id}/print">打印报告 ↗</a></div>`;
  return html;
}
async function submitReview(event) {
  event.preventDefault(); if(state.busy) return;
  let payload;
  try {
    payload={...config(), title:$('title').value, studentName:$('student').value, workContent:$('work').value, answerContent:$('answer').value, rubric:rubricValue()};
    if(state.mode==='batch') {
      payload.useDeepseek=false;
      payload.items=$('batch-text').value.split(/\r?\n/).filter(s=>s.trim()).map((line,i)=>{
        const split=line.indexOf('|');if(split<1) throw new Error(`第 ${i+1} 行缺少姓名或 | 分隔符。`);
        return {studentName:line.slice(0,split).trim(),workContent:line.slice(split+1).trim()};
      });
    }
  } catch(e){toast(e.message,true);return;}
  const mode=state.mode;
  lockWorkspace(true);$('run-grade').classList.add('busy');$('result-state').textContent='批改中…';
  try {
    const result=await api(mode==='batch'?'/api/batch':'/compare_texts',jsonOpts('POST',payload));
    if(mode==='batch') {
      $('result-content').innerHTML=`<div class="empty-result"><h3>已完成 ${result.count} 份批改</h3><p>全部记录已保存，可前往批改记录复核。</p><a class="button secondary" href="#history">查看批改记录</a></div>`;
      toast(`已完成 ${result.count} 份作业的批改。`);
    } else {state.result=result;$('result-content').innerHTML=resultMarkup(result);toast('批改完成，记录已保存。');}
    $('result-empty').hidden=true;$('result-content').hidden=false;$('result-state').textContent='已完成';
    refreshStats().catch(e=>toast('记录已保存，统计刷新失败：'+e.message,true));
  } catch(e) {toast(e.message,true);$('result-state').textContent='未完成';}
  finally {lockWorkspace(false);$('run-grade').classList.remove('busy');}
}
function preview(kind) {
  const input=$('file-'+kind),file=input.files[0],img=$('preview-'+kind),box=$('drop-'+kind);
  if(state.urls[kind]) URL.revokeObjectURL(state.urls[kind]);
  delete state.urls[kind];img.hidden=true;img.removeAttribute('src');
  box.querySelector('b').textContent=kind==='work'?'上传学生作业':'上传参考答案';
  if(file){
    if(file.size>8*1024*1024 || !/\.(jpe?g|png|bmp|tiff?|webp)$/i.test(file.name)){input.value='';toast('请选择不超过 8 MB 的支持格式图片。',true);return;}
    const url=URL.createObjectURL(file);state.urls[kind]=url;img.src=url;img.hidden=false;box.querySelector('b').textContent=file.name;
  }
  invalidateResult();
}
async function recognize() {
  if(state.busy) return;
  if(!$('file-work').files[0]||!$('file-answer').files[0]){toast('请先选择两张图片。',true);return;}
  const data=new FormData();data.append('file1',$('file-work').files[0]);data.append('file2',$('file-answer').files[0]);data.append('model',$('ocr-engine').value);data.append('language',$('language').value);
  data.append('preprocessing',$('preprocessing').value);
  lockWorkspace(true);$('run-ocr').classList.add('busy');
  try {const result=await api('/ocr',{method:'POST',body:data});$('work').value=result.workContent;$('answer').value=result.answerContent;baseline();toast(result.warnings.length?result.warnings.join(' '):'识别完成，请校对文本后开始批改。',!!result.warnings.length);}
  catch(e){toast(e.message,true);}finally{lockWorkspace(false);$('run-ocr').classList.remove('busy');}
}
function historyQuery() {return new URLSearchParams({q:$('history-search').value,status:$('history-status').value,page:state.page,pageSize:10});}
let historyRequest=0;
async function loadHistory() {
  const version=++historyRequest;const params=historyQuery();
  const data=await api('/api/records?'+params);if(version!==historyRequest)return;
  state.total=data.total;
  if(data.items.length===0 && state.page>1){state.page=Math.max(1,Math.ceil(data.total/10));return loadHistory();}
  $('history-body').innerHTML=data.items.length?data.items.map(r=>`<tr><td>${esc(r.studentName)}<small>${esc(r.title)}</small></td><td><b>${esc(r.score)}</b> / ${esc(r.options.maxScore)}</td><td><span class="result-tag ${r.passed?'':'review'}">${r.passed?'已达标':'待提升'}</span></td><td>${esc(new Date(r.createdAt).toLocaleString('zh-CN',{hour12:false}))}</td><td><div class="table-actions"><button data-view="${esc(r.id)}">查看</button><button class="delete" data-delete="${esc(r.id)}">删除</button></div></td></tr>`).join(''):'<tr><td colspan="5"><div class="empty-state">没有符合条件的批改记录。<br>回到工作台，开始第一份批改吧。</div></td></tr>';
  $('history-total').textContent=`共 ${data.total} 份 · 第 ${state.page} / ${Math.max(1,Math.ceil(data.total/10))} 页`;
  $('prev-page').disabled=state.page<=1;$('next-page').disabled=state.page*10>=data.total;
  params.delete('page');params.delete('pageSize');$('export-all').href='/api/export?'+params;
}
async function loadTemplates() {
  const data=await api('/api/templates');state.templates=data.items;
  const old=$('template-select').value;
  $('template-select').innerHTML='<option value="">从模板快速填入</option>'+data.items.map(t=>`<option value="${esc(t.id)}">${esc(t.title)}</option>`).join('');
  if(data.items.some(t=>t.id===old))$('template-select').value=old;
  $('template-cards').innerHTML=data.items.length?data.items.map(t=>`<article class="panel template-card"><span class="stat-icon mint"><svg><use href="#i-book"/></svg></span><h3>${esc(t.title)}</h3><p>${esc(t.answerContent)}</p><div class="actions"><button class="button secondary" data-use="${esc(t.id)}">使用模板</button><button class="text-button" data-edit="${esc(t.id)}">编辑</button><button class="text-button" data-remove-template="${esc(t.id)}">删除</button></div></article>`).join(''):'<div class="panel empty-state">模板库还是空的。<br>新建一个模板，保存常用参考答案与评分要点。</div>';
}
function useTemplate(id) {
  if(state.busy){toast('请等待当前批改完成。');return;}
  const t=state.templates.find(t=>t.id===id);if(!t)return;
  $('title').value=t.title;$('answer').value=t.answerContent;$('rubric').value=rubricText(t.rubric);applyConfig(t.options);
  $('template-select').value=t.id;
  state.original.answer=t.answerContent;counts();invalidateResult();location.hash='workspace';toast('已填入模板答案与评分规则。');
}
function openTemplate(id) {
  const t=state.templates.find(t=>t.id===id);
  $('template-id').value=t?.id||'';$('template-title').value=t?.title||'';$('template-answer').value=t?.answerContent||'';$('template-rubric').value=rubricText(t?.rubric);
  $('template-dialog-title').textContent=t?'编辑答案模板':'新建答案模板';$('template-dialog').showModal();
}
async function saveTemplate(event) {
  event.preventDefault();const submit=event.submitter;if(submit.disabled)return;
  try {
    const id=$('template-id').value;const old=state.templates.find(t=>t.id===id);
    const data={title:$('template-title').value,answerContent:$('template-answer').value,rubric:rubricValue('template-rubric'),options:old?.options||config()};
    submit.disabled=true;
    await api('/api/templates'+(id?'/'+encodeURIComponent(id):''),jsonOpts(id?'PUT':'POST',data));
    $('template-dialog').close();await Promise.all([loadTemplates(),refreshStats()]);toast('答案模板已保存。');
  }catch(e){toast(e.message,true);}finally{submit.disabled=false;}
}
async function health() {
  const data=await api('/api/health');
  const rows=[['文本基线 / 记录管理',true,'可用'],...Object.entries(data.ocr).map(([k,v])=>[k,v,data.ocrDetails?.[k] || (v?'依赖已安装':'未配置')]),['本地 Transformer',data.transformer.installed&&data.transformer.weights,data.transformer.installed?(data.transformer.weights?'依赖与权重已找到':'缺少模型权重'):'未安装 Paddle'],['DeepSeek',data.deepseek,data.deepseek?'密钥已配置 · 未验证连接':'未配置密钥']];
  $('health-list').innerHTML=rows.map(([name,ok,note])=>`<div class="health-item"><b>${esc(name)}</b><span class="${ok?'ready':''}">${ok?'●':'○'} ${esc(note)}</span></div>`).join('');
}
async function loadSettings() {
  const s=await api('/api/settings');applyConfig(s);
  $('setting-engine').value=s.engine;$('setting-max').value=s.maxScore;$('setting-pass').value=s.passPercent;$('setting-weight').value=s.aiWeight;$('setting-ai').checked=s.useDeepseek;
}
async function route() {
  const names={workspace:'批改工作台',history:'批改记录',templates:'答案模板',analytics:'学习概览',settings:'评分与环境'};
  let page=location.hash.slice(1)||'workspace';if(!names[page])page='workspace';
  document.querySelectorAll('.page').forEach(el=>el.hidden=el.id!=='page-'+page);
  document.querySelectorAll('[data-page]').forEach(el=>{el.classList.toggle('active',el.dataset.page===page);if(el.dataset.page===page)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  $('breadcrumb').textContent=names[page];
  try {if(page==='history')await loadHistory();if(page==='templates')await loadTemplates();if(page==='analytics')await refreshStats();if(page==='settings')await health();}
  catch(e){toast(e.message,true);}
}
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
$('review-form').addEventListener('submit',submitReview);
$('run-ocr').addEventListener('click',recognize);
$('ai').addEventListener('change',aiVisibility);$('ai-weight').addEventListener('input',aiVisibility);
document.querySelectorAll('#page-workspace input,#page-workspace textarea,#page-workspace select').forEach(el=>el.addEventListener('input',()=>{counts();invalidateResult();}));
for(const kind of ['work','answer']) {
  $('undo-'+kind).addEventListener('click',()=>{$(kind).value=state.original[kind];counts();invalidateResult();});
  $('file-'+kind).addEventListener('change',()=>preview(kind));
  const drop=$('drop-'+kind);
  drop.addEventListener('dragover',e=>{e.preventDefault();if(!state.busy)drop.classList.add('dragging');});
  drop.addEventListener('dragleave',()=>drop.classList.remove('dragging'));
  drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('dragging');if(state.busy)return;const file=e.dataTransfer.files[0];if(file){const transfer=new DataTransfer();transfer.items.add(file);$('file-'+kind).files=transfer.files;preview(kind);}});
}
$('clear-images').addEventListener('click',()=>{for(const kind of ['work','answer']){$('file-'+kind).value='';preview(kind);}});
$('demo-images').addEventListener('click',async()=>{
  if(state.busy)return;
  lockWorkspace(true);
  try {
    const images=await Promise.all(['work','answer'].map(async kind=>{
      const response=await fetch('/api/demo-images/'+kind);
      if(!response.ok)throw new Error('演示图片不存在，请检查 examples 目录。');
      return {kind,blob:await response.blob()};
    }));
    for(const {kind,blob} of images){const transfer=new DataTransfer();transfer.items.add(new File([blob],`demo-${kind}.png`,{type:'image/png'}));$('file-'+kind).files=transfer.files;preview(kind);}
    $('title').value='软件测试基础 · 图片演示';$('student').value='图片示例同学';$('ocr-engine').value='RapidOCR';
    toast('已载入两张演示图片，点击“识别两张图片”。');
  }catch(e){toast(e.message,true);}finally{lockWorkspace(false);}
});
$('load-demo').addEventListener('click',()=>{
  setMode('text');$('title').value='软件测试 · 测试的目的';$('student').value='示例同学';
  $('answer').value='软件测试的目的是发现软件缺陷，验证软件是否满足需求，并为软件质量评估提供依据。';
  $('work').value='软件测试通过运行和检查程序发现软件缺陷，验证软件是否满足需求。';
  $('rubric').value='发现软件缺陷 | 2\n满足需求 | 2\n质量评估 | 1';
  applyConfig({engine:'lexical',maxScore:100,passPercent:60,useDeepseek:false,aiWeight:70});baseline();
  toast('演示文本已填入，点击“开始批改”生成真实结果。');
});
$('template-select').addEventListener('change',()=>useTemplate($('template-select').value));
$('history-search').addEventListener('input',()=>{state.page=1;clearTimeout(searchTimer);searchTimer=setTimeout(()=>loadHistory().catch(e=>toast(e.message,true)),250);});
$('history-status').addEventListener('change',()=>{state.page=1;loadHistory().catch(e=>toast(e.message,true));});
$('prev-page').addEventListener('click',()=>{state.page--;loadHistory().catch(e=>toast(e.message,true));});
$('next-page').addEventListener('click',()=>{state.page++;loadHistory().catch(e=>toast(e.message,true));});
$('history-body').addEventListener('click',async e=>{
  const b=e.target.closest('button');if(!b)return;
  try {
    if(b.dataset.view){const r=await api('/api/records/'+encodeURIComponent(b.dataset.view));$('record-detail').innerHTML=resultMarkup(r,true);$('record-dialog').showModal();}
    if(b.dataset.delete && confirm('删除这份批改记录？删除后将不再计入统计。')){b.disabled=true;await api('/api/records/'+encodeURIComponent(b.dataset.delete),{method:'DELETE'});await Promise.all([loadHistory(),refreshStats()]);if(state.result?.id===b.dataset.delete)invalidateResult();toast('记录已删除。');}
  }catch(err){b.disabled=false;toast(err.message,true);}
});
$('template-cards').addEventListener('click',async e=>{
  const b=e.target.closest('button');if(!b)return;
  if(b.dataset.use)useTemplate(b.dataset.use);if(b.dataset.edit)openTemplate(b.dataset.edit);
  if(b.dataset.removeTemplate && confirm('删除这个答案模板？已保存的批改记录不受影响。')){
    b.disabled=true;try{await api('/api/templates/'+encodeURIComponent(b.dataset.removeTemplate),{method:'DELETE'});await Promise.all([loadTemplates(),refreshStats()]);toast('模板已删除。');}catch(err){b.disabled=false;toast(err.message,true);}
  }
});
$('new-template').addEventListener('click',()=>openTemplate());$('template-form').addEventListener('submit',saveTemplate);
document.querySelectorAll('.close-dialog').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
$('settings-form').addEventListener('submit',async e=>{
  e.preventDefault();if(state.busy){toast('请等待当前批改完成后再保存设置。');return;}
  const b=e.submitter;b.disabled=true;
  try {const s=await api('/api/settings',jsonOpts('PUT',{engine:$('setting-engine').value,maxScore:Number($('setting-max').value),passPercent:Number($('setting-pass').value),aiWeight:Number($('setting-weight').value),useDeepseek:$('setting-ai').checked}));applyConfig(s);invalidateResult();toast('默认规则已保存。');}
  catch(err){toast(err.message,true);}finally{b.disabled=false;}
});
$('refresh-health').addEventListener('click',()=>health().then(()=>toast('环境状态已刷新。')).catch(e=>toast(e.message,true)));
try {document.documentElement.dataset.theme=localStorage.getItem('homework-theme')||'light';}catch{}
$('theme-toggle').addEventListener('click',()=>{const t=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=t;try{localStorage.setItem('homework-theme',t);}catch{}});
window.addEventListener('hashchange',route);
window.addEventListener('beforeunload',e=>{if(state.busy){e.preventDefault();e.returnValue='';}});
Promise.allSettled([loadSettings(),loadTemplates(),refreshStats(),health()]).then(results=>{const errors=results.filter(r=>r.status==='rejected');if(errors.length)toast('部分数据加载失败：'+errors[0].reason.message,true);route();});
