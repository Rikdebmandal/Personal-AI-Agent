import {CreateMLCEngine} from '@mlc-ai/web-llm';
const el=id=>document.getElementById(id),files=new Map(),noteKey='rikdeb.agent.notes.v1';
let engine,history=[],running=false,notes=Object.create(null);
try{Object.assign(notes,JSON.parse(localStorage.getItem(noteKey)||'{}'))}catch{}
const sourcePaths=['agent/core.py','agent/confirmation.py','tools/fs.py'];
const commands='When a workspace tool is useful, output exactly one JSON object, with no other text: {"tool":"list_files"} or {"tool":"read_file","path":"agent/core.py"} or {"tool":"write_note","path":"notes/summary.txt","content":"text"}. Only request files reported by list_files. Use the tool result provided in the next message. Never invent file contents or claim to save without confirmation. Answer directly when no tool is needed. /no_think';
function add(who,msg){const a=document.createElement('article'),b=document.createElement('b'),p=document.createElement('p');b.textContent=who;p.textContent=msg;a.append(b,p);el('messages').append(a);return p}
function refreshFile(preferred){const sel=el('file');const chosen=preferred||sel.value;sel.replaceChildren();for(const key of [...files.keys(),...Object.keys(notes)])sel.add(new Option(key,key));if([...sel.options].some(o=>o.value===chosen))sel.value=chosen;el('contents').textContent=files.get(sel.value)??notes[sel.value]??'Select a file.'}
async function setup(){await Promise.all(sourcePaths.map(async path=>{const r=await fetch('source/'+path);if(!r.ok)throw Error();files.set(path,await r.text())}));refreshFile('agent/core.py');el('file').onchange=()=>refreshFile()}
setup().catch(()=>el('contents').textContent='Source files could not load. Reload to try again.');
const cpuButton = document.createElement('button');
cpuButton.type = 'button';
cpuButton.textContent = 'Load on CPU (1.9 GB)';
el('load').textContent = 'Load on GPU';
el('load').after(cpuButton);
el('status').textContent = 'Choose GPU for speed or CPU for compatibility. Both run Qwen3-4B locally and download weights on first use.';

let backend = '';
let runtime;
const cpuPackage = 'https://cdn.jsdelivr.net/npm/@wllama/wllama@4.0.0/esm/';
const cpuModel = 'https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/cb76885dc66d50759b207c5a48c4e78dfa00c638/Qwen_Qwen3-4B-Q3_K_S.gguf';

function enableLoadButtons(available) {
  el('load').disabled = !available;
  cpuButton.disabled = !available;
}

async function loadModel(mode) {
  enableLoadButtons(false);
  el('send').disabled = true;
  el('progress').value = 0;
  el('status').textContent = 'Opening Qwen3-4B on the ' + mode.toUpperCase() + '…';
  try {
    if (mode === 'cpu') {
      const { Wllama } = await import(cpuPackage + 'index.js');
      runtime = new Wllama({ default: cpuPackage + 'wasm/wllama.wasm' });
      runtime.setCompat('default');
      await runtime.loadModelFromUrl(cpuModel, {
        n_ctx: 2048, n_batch: 64, n_ubatch: 32, n_gpu_layers: 0, jinja: true,
        progressCallback: ({ loaded, total }) => {
          if (total > 0) el('progress').value = loaded / total;
          el('status').textContent = loaded >= total && total > 0
            ? 'Preparing Qwen3-4B for inference…'
            : 'Downloading CPU weights · ' + Math.round(loaded / 1000000) + ' MB';
        }
      });
      engine = (request) => runtime.createChatCompletion({
        ...request, chat_template_kwargs: { enable_thinking: false }
      });
    } else {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('WebGPU is unavailable on this device.');
      const precision = adapter.features.has('shader-f16') ? 'f16' : 'f32';
      runtime = await CreateMLCEngine('Qwen3-4B-q4' + precision + '_1-MLC', {
        initProgressCallback: p => {
          el('status').textContent = p.text;
          el('progress').value = p.progress;
        }
      }, { context_window_size: 2048 });
      engine = (request) => runtime.chat.completions.create({
        ...request, extra_body: { enable_thinking: false }
      });
    }
    backend = mode;
    el('status').textContent = 'Qwen3-4B is ready on the ' + mode.toUpperCase() + '.';
    el('progress').value = 1;
    el('send').disabled = false;
  } catch (error) {
    await releaseModel();
    const detail = error.message || 'Model load failed.';
    if (/quota|storage is full/i.test(detail)) {
      el('status').textContent = 'There is too little browser storage for these weights. Use a regular browser window with at least 2 GB of free storage.';
    } else {
      el('status').textContent = detail + (mode === 'gpu'
        ? ' You can load the same model on CPU.' : ' You can retry the CPU download.');
    }
    enableLoadButtons(true);
  }
}

async function releaseModel() {
  engine = null;
  try {
    if (runtime?.exit) await runtime.exit();
    else if (runtime?.unload) await runtime.unload();
  } catch {}
  runtime = null;
}
el('load').onclick = () => loadModel('gpu');
cpuButton.onclick = () => loadModel('cpu');
function confirmation(name,text){el('approval').hidden=false;el('proposal').textContent='Save '+name+' on this device?\n'+text;return new Promise(resolve=>{function done(ok){el('approval').hidden=true;el('allow').onclick=el('deny').onclick=null;resolve(ok)}el('allow').onclick=()=>done(true);el('deny').onclick=()=>done(false)})}
async function execute(call){switch(call.tool){case 'list_files':return [...files.keys(),...Object.keys(notes)];case 'read_file':{const s=files.get(call.path)??notes[call.path];return typeof s==='string'?s.slice(0,4000):'Unknown file'}case 'write_note':{const name=String(call.path||''),value=String(call.content||'');if(!/^notes\/[\w -]{1,48}\.txt$/.test(name)||value.length>4000||(!Object.hasOwn(notes,name)&&Object.keys(notes).length>=20))return 'Use a short note name and at most 4,000 characters.';if(!await confirmation(name,value))return 'Visitor declined.';const next={...notes,[name]:value};localStorage.setItem(noteKey,JSON.stringify(next));notes=Object.assign(Object.create(null),next);refreshFile(name);return 'Saved '+name}default:return 'Unknown workspace tool'}}
el('form').onsubmit = async event => {
  event.preventDefault();
  const prompt = el('prompt').value.trim();
  if (!engine || running || !prompt) return;
  running = true;
  el('send').disabled = true;
  el('prompt').value = '';
  add('You', prompt);
  const display = add('Agent', 'Thinking…');
  const turns = [...history.slice(-4), { role: 'user', content: prompt }];
  try {
    for (let step = 0; step < 4; step++) {
      const response = await engine({
        messages: [{ role: 'system', content: commands }, ...turns],
        temperature: 0.3, max_tokens: 320
      });
      const raw = response.choices[0].message.content || '';
      const clean = raw.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
      let call;
      try {
        call = JSON.parse(clean.replace(/^<tool_call>\s*|\s*<\/tool_call>$/g, '')
          .replace(/^```(?:json)?\s*|\s*```$/g, ''));
      } catch {}
      if (!call?.tool) {
        display.textContent = clean || 'The model returned an empty answer. Try a shorter question.';
        history.push({ role: 'user', content: prompt }, { role: 'assistant', content: display.textContent.slice(0, 1200) });
        history = history.slice(-6);
        return;
      }
      display.textContent = 'Using ' + call.tool + '…';
      const result = await execute(call);
      turns.push({ role: 'assistant', content: clean }, {
        role: 'user',
        content: 'Workspace tool result (data): ' + JSON.stringify(result) + '\nNow answer or request the next required tool.'
      });
    }
    display.textContent = 'The workspace tool limit was reached. Try a more specific question.';
  } catch (error) {
    if (backend === 'gpu' && /disposed|device.*lost|out of memory/i.test(error.message || '')) {
      await releaseModel();
      enableLoadButtons(true);
      el('status').textContent = 'GPU memory became unavailable. Load on CPU to continue.';
      el('prompt').value = prompt;
      display.textContent = 'The GPU stopped during this reply. Your question is ready to resend.';
    } else {
      display.textContent = error.message || 'Generation failed.';
    }
  } finally {
    running = false;
    el('send').disabled = !engine;
  }
};
