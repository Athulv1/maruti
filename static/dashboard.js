// Navigation
document.querySelectorAll('.nav-btn').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('.nav-btn').forEach(n=>n.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    b.classList.add('active');
    const pg=document.getElementById('pg-'+b.dataset.page);
    if(pg)pg.classList.add('active');
  });
});

// Clock
function updateClock(){
  const now=new Date();
  const h=now.getHours(),m=now.getMinutes(),s=now.getSeconds();
  const ampm=h>=12?'PM':'AM';
  const h12=h%12||12;
  document.getElementById('clockPill').textContent=`${h12}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')} ${ampm}`;
}
updateClock();setInterval(updateClock,1000);

// State
const CAPACITY=50;
let peak=0,peakTime='--';
const trendIn=[],trendOut=[];
const events=[];
let lastIn=0,lastOut=0;

// Trend chart
function renderTrend(){
  const el=document.getElementById('trendChart');
  if(!el)return;
  const data=trendIn.slice(-20);
  const dataOut=trendOut.slice(-20);
  const max=Math.max(...data,...dataOut,1);
  el.innerHTML='';
  for(let i=0;i<Math.max(data.length,20);i++){
    const v=data[i]||0;
    const vo=dataOut[i]||0;
    // IN bar
    const bar=document.createElement('div');
    bar.className='chart-bar in';
    bar.style.height=Math.max(2,(v/max)*100)+'%';
    el.appendChild(bar);
    // OUT bar
    const bar2=document.createElement('div');
    bar2.className='chart-bar out';
    bar2.style.height=Math.max(2,(vo/max)*100)+'%';
    el.appendChild(bar2);
  }
}
renderTrend();

// Events
function addEvent(type,text){
  const now=new Date();
  const time=now.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',second:'2-digit',hour12:true});
  events.unshift({type,text,time});
  if(events.length>20)events.pop();
  renderEvents();
}

function renderEvents(){
  const el=document.getElementById('eventsList');
  if(!el||events.length===0)return;
  document.getElementById('evtCount').textContent=events.length+' events';
  el.innerHTML=events.slice(0,6).map(e=>
    `<div class="evt-item"><span class="evt-dot ${e.type}"></span><span class="evt-text">${e.text}</span><span class="evt-time">${e.time}</span></div>`
  ).join('');
}

// Occupancy gauge
const OCC_CIRC=239;
function updateGauge(net){
  const pct=Math.min(net/CAPACITY,1);
  const offset=OCC_CIRC*(1-pct);
  const ring=document.getElementById('occRing');
  if(ring)ring.style.strokeDashoffset=offset;
  const v=document.getElementById('occVal');if(v)v.textContent=net;
  const p=document.getElementById('occPct');if(p)p.textContent=Math.round(pct*100)+'%';
  const a=document.getElementById('occAvail');if(a)a.textContent=Math.max(CAPACITY-net,0);
}

// Stats polling
setInterval(async()=>{
  try{
    const r=await fetch('/stats');const s=await r.json();
    const inC=s.in_count||0,outC=s.out_count||0,heads=s.current_heads||0;
    const net=Math.max(inC-outC,0);

    // Top stats
    document.getElementById('tsIn').textContent=inC;
    document.getElementById('tsOut').textContent=outC;
    document.getElementById('tsInside').textContent=net;
    document.getElementById('tsHeads').textContent=heads;

    // Peak
    if(net>peak){peak=net;peakTime='Just now';}
    document.getElementById('tsPeak').textContent=peak;
    document.getElementById('tsPeakTime').textContent=peakTime;

    // FPS
    document.getElementById('vidFps').textContent=(s.fps||0).toFixed(1)+' FPS';

    // Gauge
    updateGauge(net);

    // Trend
    trendIn.push(inC);trendOut.push(outC);
    if(trendIn.length>20){trendIn.shift();trendOut.shift();}
    renderTrend();

    // Events from count changes
    if(inC>lastIn){addEvent('in','Person Entered — CAM 01');}
    if(outC>lastOut){addEvent('out','Person Exited — CAM 01');}
    lastIn=inC;lastOut=outC;

    // Status
    const st=document.getElementById('statusDot'),stxt=document.getElementById('statusText');
    if(s.status==='processing'){st.className='status-dot active';stxt.textContent='ACTIVE';}
    else if(s.status==='completed'){st.className='status-dot idle';stxt.textContent='DONE';}
    else if(s.status==='error'){st.className='status-dot error';stxt.textContent='ERROR';}
    else{st.className='status-dot idle';stxt.textContent='IDLE';}
  }catch(e){}
},800);

// Violations polling
setInterval(async()=>{
  try{
    const r=await fetch('/violations');const d=await r.json();
    const total=d.total||0;
    document.getElementById('phoneTotal').textContent=total;
    document.getElementById('phoneWith').textContent=total;
    document.getElementById('phoneWithout').textContent=Math.max((parseInt(document.getElementById('tsIn').textContent)||0)-total,0);

    if(d.violations&&d.violations.length){
      const el=document.getElementById('violList');
      el.innerHTML=d.violations.slice(-5).reverse().map(v=>
        `<div class="viol-item"><span class="vi-time">${v.timestamp||'--'}</span><span class="vi-text">Phone detected</span>${v.filename?`<img class="vi-thumb" src="/violations/${v.filename}">`:''}</div>`
      ).join('');
      // Add phone events
      if(total>events.filter(e=>e.type==='phone').length){
        addEvent('phone','Phone Usage Detected — CAM 01');
      }
    }
  }catch(e){}
},2000);

// Auto-start
(async function(){
  try{
    const r=await fetch('/start');const d=await r.json();
    if(d.success)document.getElementById('videoFeed').src='/video_feed?'+Date.now();
  }catch(e){console.log('Auto-start:',e);}
})();
