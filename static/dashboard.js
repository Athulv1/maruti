// Returns today's date as YYYY-MM-DD in local timezone (not UTC)
function localToday(){const d=new Date();return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;}

// Delete a violation
function deleteViolation(filename) {
  if (!confirm('Delete this violation image?')) return;
  fetch('/violations/' + filename, { method: 'DELETE' })
    .then(r => r.json())
    .then(d => {
      if (d.success) {
        // Remove from live panel
        const liveEl = document.getElementById('vi-' + filename);
        if (liveEl) liveEl.remove();
        // Remove from reports table
        const rowEl = document.getElementById('vrow-' + filename);
        if (rowEl) rowEl.remove();
        // Refresh both live counts and reports summary immediately
        refreshViolationCounts();
        refreshReports();
      } else {
        alert('Delete failed: ' + (d.error || 'Unknown error'));
      }
    })
    .catch(() => alert('Delete request failed'));
}

function refreshViolationCounts() {
  const today = localToday();
  fetch('/violations?date=' + today).then(r => r.json()).then(d => {
    const total = d.total || 0;
    if (phoneBaseline > total) phoneBaseline = total;
    document.getElementById('phoneTotal').textContent = total;
    document.getElementById('phoneWith').textContent = total;
    document.getElementById('phoneWithout').textContent = Math.max((parseInt(document.getElementById('tsIn').textContent) || 0) - total, 0);
  }).catch(() => {});
}

// Reset counts
function resetCounts() {
  if (!confirm('Reset IN and OUT counts to zero?')) return;
  fetch('/reset_counts', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        const inEl = document.getElementById('tsIn');
        const outEl = document.getElementById('tsOut');
        if (inEl) inEl.textContent = '0';
        if (outEl) outEl.textContent = '0';
      } else {
        alert('Reset failed: ' + (data.message || 'Unknown error'));
      }
    })
    .catch(() => alert('Reset request failed'));
}

// Navigation
document.querySelectorAll('.nav-btn').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('.nav-btn').forEach(n=>n.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    b.classList.add('active');
    const pg=document.getElementById('pg-'+b.dataset.page);
    if(pg)pg.classList.add('active');
    if(b.dataset.page==='settings') loadRoiFrame();
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
let lastRestartAttempt=0;
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
    else if(s.status==='completed'||s.status==='error'){
      st.className='status-dot '+(s.status==='error'?'error':'idle');
      stxt.textContent=s.status==='error'?'ERROR':'DONE';
      // Auto-restart if stream stopped unexpectedly
      const now=Date.now();
      if(now-lastRestartAttempt>30000){
        lastRestartAttempt=now;
        fetch('/start').then(r=>r.json()).then(d=>{
          if(d.success)document.getElementById('videoFeed').src='/video_feed?'+Date.now();
        }).catch(()=>{});
      }
    }
    else{st.className='status-dot idle';stxt.textContent='IDLE';}
  }catch(e){}
},800);

// Violations polling
let phoneBaseline = -1; // violations already on disk when page loaded — don't fire events for these
setInterval(async()=>{
  try{
    const today=localToday();
    const r=await fetch('/violations?date='+today);const d=await r.json();
    const total=d.total||0;
    if(phoneBaseline===-1) phoneBaseline=total; // snapshot count on first poll
    document.getElementById('phoneTotal').textContent=total;
    document.getElementById('phoneWith').textContent=total;
    document.getElementById('phoneWithout').textContent=Math.max((parseInt(document.getElementById('tsIn').textContent)||0)-total,0);

    const el=document.getElementById('violList');
    if(d.violations&&d.violations.length){
      el.innerHTML=d.violations.slice(-5).reverse().map(v=>
        `<div class="viol-item" id="vi-${v.filename}">
          <span class="vi-time">${v.timestamp||'--'}</span>
          <span class="vi-text">Phone detected</span>
          ${v.filename?`<img class="vi-thumb" src="/violations/${v.filename}" onclick="window.open('/violations/${v.filename}','_blank')" style="cursor:pointer">`:'' }
          ${v.filename?`<button onclick="deleteViolation('${v.filename}')" style="margin-left:auto;background:none;border:none;cursor:pointer;color:#EF4444;font-size:14px;padding:2px 4px;flex-shrink:0" title="Delete">✕</button>`:''}
        </div>`
      ).join('');
    }
    // Only fire event for violations detected AFTER page load
    const newSinceLoad = total - phoneBaseline;
    if(newSinceLoad > 0 && newSinceLoad > events.filter(e=>e.type==='phone').length){
      addEvent('phone','Phone Usage Detected — CAM 01');
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

// Reports page

// Excel Download helpers
function saveExcel(wb, filename){
  if(typeof XLSX==='undefined'){alert('Excel library not loaded. Please refresh the page and try again.');return;}
  XLSX.writeFile(wb, filename);
}

function checkXLSX(){
  if(typeof XLSX==='undefined'){alert('Excel library not loaded. Please refresh the page and try again.');return false;}
  return true;
}

function downloadDailyCSV(){
  if(!checkXLSX())return;
  const dp=document.getElementById('reportDate');
  const dateStr=dp?dp.value:localToday();
  fetch('/api/reports/'+dateStr).then(r=>r.json()).then(d=>{
    if(d.error){alert('No data available for '+dateStr);return;}
    const wb=XLSX.utils.book_new();
    // Hourly sheet
    const hourlyRows=[['Hour','IN Count','OUT Count','Net Inside','Violations']];
    if(d.hourly){
      d.hourly.forEach(h=>{
        hourlyRows.push([h.label, h.in_count, h.out_count, Math.max(h.in_count-h.out_count,0), h.violations]);
      });
    }
    const ws1=XLSX.utils.aoa_to_sheet(hourlyRows);
    ws1['!cols']=[{wch:8},{wch:12},{wch:12},{wch:12},{wch:12}];
    XLSX.utils.book_append_sheet(wb, ws1, 'Hourly Breakdown');
    // Summary sheet
    const summaryRows=[
      ['Metric','Value'],
      ['Date', dateStr],
      ['Total IN', d.summary.total_in],
      ['Total OUT', d.summary.total_out],
      ['Net Inside', Math.max(d.summary.total_in-d.summary.total_out,0)],
      ['Total Violations', d.summary.total_violations],
      ['Peak Hour', d.summary.peak_hour]
    ];
    const ws2=XLSX.utils.aoa_to_sheet(summaryRows);
    ws2['!cols']=[{wch:18},{wch:14}];
    XLSX.utils.book_append_sheet(wb, ws2, 'Summary');
    saveExcel(wb, `Daily_Report_${dateStr}.xlsx`);
  }).catch(()=>alert('Failed to fetch report data'));
}

function downloadWeeklyCSV(){
  const a = document.createElement('a');
  a.href = '/api/reports/weekly_pdf';
  a.download = `Weekly_Report_${localToday()}.pdf`;
  a.click();
}

function downloadViolationsCSV(){
  if(!checkXLSX())return;
  const dp=document.getElementById('reportDate');
  const dateStr=dp?dp.value:localToday();
  fetch('/violations?date='+dateStr).then(r=>r.json()).then(d=>{
    if(!d.violations||d.violations.length===0){alert('No violations recorded for '+dateStr);return;}
    const wb=XLSX.utils.book_new();
    const rows=[['#','Timestamp','Frame Number','Screenshot File']];
    d.violations.forEach((v,i)=>{
      rows.push([i+1, v.timestamp||'--', v.frame_number||'--', v.filename||'--']);
    });
    rows.push([]);
    rows.push(['Total Violations', d.violations.length]);
    const ws=XLSX.utils.aoa_to_sheet(rows);
    ws['!cols']=[{wch:5},{wch:22},{wch:14},{wch:35}];
    XLSX.utils.book_append_sheet(wb, ws, 'Violations');
    saveExcel(wb, `Violations_Report_${dateStr}.xlsx`);
  }).catch(()=>alert('Failed to fetch violations'));
}

let rptHourlyChart=null, rptViolChart=null, rptWeeklyChart=null;

const chartFont={family:'Inter',size:10};
const chartGrid={color:'#F1F5F9'};
const chartTicks={font:{size:9},color:'#9CA3AF'};

function initReportCharts(){
  if(rptHourlyChart)return;
  const ctx1=document.getElementById('rptHourlyChart');
  const ctx2=document.getElementById('rptViolChart');
  const ctx3=document.getElementById('rptWeeklyChart');
  if(!ctx1||!ctx2||!ctx3)return;

  // Hourly traffic line chart
  rptHourlyChart=new Chart(ctx1,{
    type:'line',
    data:{labels:[],datasets:[
      {label:'IN (Cumulative)',data:[],borderColor:'#2563EB',backgroundColor:'rgba(37,99,235,0.08)',fill:true,tension:.4,pointRadius:3,borderWidth:2},
      {label:'OUT (Cumulative)',data:[],borderColor:'#EF4444',backgroundColor:'rgba(239,68,68,0.08)',fill:true,tension:.4,pointRadius:3,borderWidth:2}
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{boxWidth:10,font:chartFont,usePointStyle:true,padding:12}}},
      scales:{x:{grid:{display:false},ticks:chartTicks},y:{beginAtZero:true,grid:chartGrid,ticks:chartTicks}},
      animation:{duration:500}
    }
  });

  // Hourly violations bar chart
  rptViolChart=new Chart(ctx2,{
    type:'bar',
    data:{labels:[],datasets:[
      {label:'Violations',data:[],backgroundColor:'rgba(234,88,12,0.7)',borderColor:'#EA580C',borderWidth:1,borderRadius:4}
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{x:{grid:{display:false},ticks:chartTicks},y:{beginAtZero:true,grid:chartGrid,ticks:{...chartTicks,stepSize:1}}},
      animation:{duration:500}
    }
  });

  // Weekly overview grouped bar chart
  rptWeeklyChart=new Chart(ctx3,{
    type:'bar',
    data:{labels:[],datasets:[
      {label:'Total IN',data:[],backgroundColor:'rgba(37,99,235,0.7)',borderColor:'#2563EB',borderWidth:1,borderRadius:4},
      {label:'Total OUT',data:[],backgroundColor:'rgba(239,68,68,0.6)',borderColor:'#EF4444',borderWidth:1,borderRadius:4},
      {label:'Violations',data:[],backgroundColor:'rgba(234,88,12,0.6)',borderColor:'#EA580C',borderWidth:1,borderRadius:4}
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{boxWidth:10,font:chartFont,usePointStyle:true,padding:12}}},
      scales:{x:{grid:{display:false},ticks:chartTicks},y:{beginAtZero:true,grid:chartGrid,ticks:chartTicks}},
      animation:{duration:500}
    }
  });
}

function loadDailyReport(dateStr){
  initReportCharts();
  fetch('/api/reports/'+dateStr).then(r=>r.json()).then(d=>{
    if(d.error)return;
    // Update summary cards from historical data
    if(d.summary){
      const ri=document.getElementById('rptIn');if(ri)ri.textContent=d.summary.total_in;
      const ro=document.getElementById('rptOut');if(ro)ro.textContent=d.summary.total_out;
      const rn=document.getElementById('rptNet');if(rn)rn.textContent=Math.max(d.summary.total_in-d.summary.total_out,0);
      // rptViolations is set from the file list in refreshReports, not from DB, so deletions are reflected
      // Table
      const ti=document.getElementById('rptTblIn');if(ti)ti.textContent=d.summary.total_in;
      const to2=document.getElementById('rptTblOut');if(to2)to2.textContent=d.summary.total_out;
      const tn=document.getElementById('rptTblNet');if(tn)tn.textContent=Math.max(d.summary.total_in-d.summary.total_out,0);
    }

    // Update hourly charts
    if(d.hourly&&rptHourlyChart){
      const labels=d.hourly.map(h=>h.label);
      rptHourlyChart.data.labels=labels;
      rptHourlyChart.data.datasets[0].data=d.hourly.map(h=>h.in_count);
      rptHourlyChart.data.datasets[1].data=d.hourly.map(h=>h.out_count);
      rptHourlyChart.update();

      rptViolChart.data.labels=labels;
      rptViolChart.data.datasets[0].data=d.hourly.map(h=>h.violations);
      rptViolChart.update();
    }
  }).catch(()=>{});
}

function loadWeeklyReport(){
  initReportCharts();
  fetch('/api/reports/week').then(r=>r.json()).then(d=>{
    if(d.days&&rptWeeklyChart){
      rptWeeklyChart.data.labels=d.days.map(x=>x.day_name+' '+x.date.slice(5));
      rptWeeklyChart.data.datasets[0].data=d.days.map(x=>x.total_in);
      rptWeeklyChart.data.datasets[1].data=d.days.map(x=>x.total_out);
      rptWeeklyChart.data.datasets[2].data=d.days.map(x=>x.total_violations);
      rptWeeklyChart.update();
    }
  }).catch(()=>{});
}

function openClientReport(){
  const dp=document.getElementById('reportDate');
  const dateStr=dp?dp.value:localToday();
  window.open('/report/'+dateStr,'_blank');
}

function refreshReports(){
  const dp=document.getElementById('reportDate');
  const dateStr=dp?dp.value:localToday();
  loadDailyReport(dateStr);

  // Also update live stats for today
  const today=localToday();
  if(dateStr===today){
    fetch('/stats').then(r=>r.json()).then(s=>{
      const inC=s.in_count||0,outC=s.out_count||0,net=Math.max(inC-outC,0),heads=s.current_heads||0;
      const ri=document.getElementById('rptIn');if(ri)ri.textContent=inC;
      const ro=document.getElementById('rptOut');if(ro)ro.textContent=outC;
      const rn=document.getElementById('rptNet');if(rn)rn.textContent=net;
      const ti=document.getElementById('rptTblIn');if(ti)ti.textContent=inC;
      const to2=document.getElementById('rptTblOut');if(to2)to2.textContent=outC;
      const tn=document.getElementById('rptTblNet');if(tn)tn.textContent=net;
      const th=document.getElementById('rptTblHeads');if(th)th.textContent=heads;
    }).catch(()=>{});
  }

  // Fetch violations for table (filtered by selected date)
  fetch('/violations?date='+dateStr).then(r=>r.json()).then(d=>{
    const total=d.total||0;
    const rc=document.getElementById('rptViolCount');if(rc)rc.textContent=total+' violations recorded';
    // Override DB value with actual file count so it's always accurate
    const rv=document.getElementById('rptViolations');if(rv)rv.textContent=total;
    const body=document.getElementById('rptViolBody');
    if(body){
      if(d.violations&&d.violations.length>0){
        body.innerHTML=d.violations.map((v,i)=>
          `<tr id="vrow-${v.filename}">
            <td style="padding:8px 12px;border-bottom:1px solid #F1F5F9;color:#6B7280">${i+1}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #F1F5F9;color:#1E293B;font-weight:500">${v.timestamp||'--'}</td>
            <td style="padding:8px 12px;text-align:center;border-bottom:1px solid #F1F5F9;color:#6B7280">${v.frame_number||'--'}</td>
            <td style="padding:8px 12px;text-align:center;border-bottom:1px solid #F1F5F9">
              ${v.filename?`<img src="/violations/${v.filename}" style="width:40px;height:30px;object-fit:cover;border-radius:4px;border:1px solid #E5E7EB;cursor:pointer" onclick="window.open('/violations/${v.filename}','_blank')">`:'-'}
              ${v.gif?`<a href="/violations/${v.gif}" target="_blank" style="margin-left:6px;font-size:10px;font-weight:600;color:#7C3AED;background:#F3E8FF;padding:2px 7px;border-radius:4px;border:1px solid #DDD6FE;text-decoration:none">GIF</a>`:''}
            </td>
            <td style="padding:8px 12px;text-align:center;border-bottom:1px solid #F1F5F9">
              ${v.filename?`<button onclick="deleteViolation('${v.filename}')" style="background:#FEF2F2;border:1px solid #FECACA;color:#EF4444;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:600;cursor:pointer" onmouseover="this.style.background='#FEE2E2'" onmouseout="this.style.background='#FEF2F2'">Delete</button>`:''}
            </td>
          </tr>`
        ).join('');
      } else {
        body.innerHTML='<tr><td colspan="4" style="padding:20px;text-align:center;color:#9CA3AF">No violations recorded yet</td></tr>';
      }
    }
  }).catch(()=>{});
}

// ── ROI Drawing ──────────────────────────────────────────────────────────────
const ROI_STEPS = [
  {label:'Step 1: Draw Upper Zone — click 4 points',         color:'#22C55E', fill:'rgba(34,197,94,0.25)',   needed:4},
  {label:'Step 2: Draw Lower Zone — click 4 points',         color:'#3B82F6', fill:'rgba(59,130,246,0.25)',  needed:4},
  {label:'Step 3: Draw Counting Line — click 2 points',      color:'#EF4444', fill:null,                    needed:2},
  {label:'Step 4 (optional): Draw Phone Zone — click to add points, double-click to finish', color:'#EA580C', fill:'rgba(234,88,12,0.25)', needed:99},
];
let roi = {step:0, upper:[], lower:[], line:[], phone:[], sx:1, sy:1};

function loadRoiFrame(){
  const img=document.getElementById('roiImg');
  if(!img) return;
  img.src='/api/roi/frame?t='+Date.now();
  img.onload=function(){
    const canvas=document.getElementById('roiCanvas');
    canvas.width=img.offsetWidth;
    canvas.height=img.offsetHeight;
    roi.sx=img.naturalWidth/img.offsetWidth;
    roi.sy=img.naturalHeight/img.offsetHeight;
    fetch('/api/roi').then(r=>r.json()).then(d=>{
      if(d.upper_box)  roi.upper=d.upper_box.map(p=>[p[0]/roi.sx, p[1]/roi.sy]);
      if(d.lower_box)  roi.lower=d.lower_box.map(p=>[p[0]/roi.sx, p[1]/roi.sy]);
      if(d.line_points)roi.line=d.line_points.map(p=>[p[0]/roi.sx, p[1]/roi.sy]);
      if(d.phone_roi)  roi.phone=d.phone_roi.map(p=>[p[0]/roi.sx, p[1]/roi.sy]);
      // Auto-advance to phone zone step if counting is already done
      const countingDone=roi.upper.length===4&&roi.lower.length===4&&roi.line.length===2;
      if(countingDone) roi.step=3;
      roiUpdateUI(); roiDraw();
    }).catch(()=>{roiUpdateUI(); roiDraw();});
  };
  img.onerror=function(){img.alt='Start the stream first, then refresh frame.';};
}

function roiCurrentPts(){return [roi.upper,roi.lower,roi.line,roi.phone][roi.step];}

document.addEventListener('click',function(e){
  const canvas=document.getElementById('roiCanvas');
  if(!canvas||e.target!==canvas) return;
  const rect=canvas.getBoundingClientRect();
  const x=e.clientX-rect.left, y=e.clientY-rect.top;
  const pts=roiCurrentPts();
  const needed=ROI_STEPS[roi.step].needed;
  if(pts.length>=needed) return;
  pts.push([x,y]);
  if(pts.length===needed && roi.step<3){roi.step++;}
  roiUpdateUI();
  roiDraw();
});

document.addEventListener('dblclick',function(e){
  const canvas=document.getElementById('roiCanvas');
  if(!canvas||e.target!==canvas) return;
  if(roi.step===3 && roi.phone.length>=3){
    // Remove the last point added by the preceding click event, then close polygon
    roi.phone.pop();
    roiUpdateUI();
    roiDraw();
  }
});

function roiUndo(){
  const pts=roiCurrentPts();
  if(pts.length>0){pts.pop(); roiDraw(); return;}
  if(roi.step>0){roi.step--; roiCurrentPts().pop(); roiUpdateUI(); roiDraw();}
}

function roiClear(){
  roi.step=0; roi.upper=[]; roi.lower=[]; roi.line=[]; roi.phone=[];
  roiUpdateUI(); roiDraw();
}

function roiUpdateUI(){
  const lbl=document.getElementById('roiStepLabel');
  const btn=document.getElementById('roiSaveBtn');
  const countingDone=roi.upper.length===4&&roi.lower.length===4&&roi.line.length===2;
  const phoneDone=roi.phone.length>=3;
  const done=countingDone; // phone zone is optional — save enabled once counting is done
  if(lbl){
    if(countingDone && phoneDone) lbl.textContent='✓ All zones set — ready to save';
    else if(countingDone) lbl.textContent='✓ Counting done — draw Phone Zone or save now';
    else lbl.textContent=ROI_STEPS[roi.step].label;
  }
  if(btn){btn.style.opacity=done?'1':'0.4'; btn.style.pointerEvents=done?'auto':'none';}
  // Step pills
  const colors=[
    {done:'#DCFCE7',doneTxt:'#16A34A',doneBdr:'#BBF7D0', act:'#EFF6FF',actTxt:'#2563EB',actBdr:'#BFDBFE'},
    {done:'#DCFCE7',doneTxt:'#16A34A',doneBdr:'#BBF7D0', act:'#EFF6FF',actTxt:'#2563EB',actBdr:'#BFDBFE'},
    {done:'#DCFCE7',doneTxt:'#16A34A',doneBdr:'#BBF7D0', act:'#EFF6FF',actTxt:'#2563EB',actBdr:'#BFDBFE'},
    {done:'#FFF7ED',doneTxt:'#EA580C',doneBdr:'#FED7AA', act:'#FFF7ED',actTxt:'#EA580C',actBdr:'#FDBA74'},
  ];
  [0,1,2,3].forEach(i=>{
    const pill=document.getElementById('roiPill'+i);
    if(!pill) return;
    const ptsDone=[roi.upper.length===4,roi.lower.length===4,roi.line.length===2,roi.phone.length>=3][i];
    if(ptsDone){pill.style.background=colors[i].done;pill.style.color=colors[i].doneTxt;pill.style.borderColor=colors[i].doneBdr;}
    else if(i===roi.step){pill.style.background=colors[i].act;pill.style.color=colors[i].actTxt;pill.style.borderColor=colors[i].actBdr;}
    else{pill.style.background='#F1F5F9';pill.style.color='#94A3B8';pill.style.borderColor='#E2E8F0';}
    pill.style.cursor='pointer';
    pill.onclick=()=>{roi.step=i; roiUpdateUI(); roiDraw();};
  });
}

function roiDraw(){
  const canvas=document.getElementById('roiCanvas');
  if(!canvas) return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // Zones
  [{pts:roi.upper,color:'#22C55E',fill:'rgba(34,197,94,0.2)'},{pts:roi.lower,color:'#3B82F6',fill:'rgba(59,130,246,0.2)'}].forEach(z=>{
    if(!z.pts.length) return;
    ctx.beginPath(); ctx.moveTo(z.pts[0][0],z.pts[0][1]);
    for(let i=1;i<z.pts.length;i++) ctx.lineTo(z.pts[i][0],z.pts[i][1]);
    if(z.pts.length===4) ctx.closePath();
    ctx.fillStyle=z.fill; ctx.fill();
    ctx.strokeStyle=z.color; ctx.lineWidth=2; ctx.stroke();
    z.pts.forEach((p,i)=>{
      ctx.beginPath(); ctx.arc(p[0],p[1],5,0,Math.PI*2);
      ctx.fillStyle=z.color; ctx.fill();
      ctx.fillStyle='#fff'; ctx.font='bold 10px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(i+1,p[0],p[1]);
    });
  });
  // Phone detection zone
  if(roi.phone.length>0){
    const z={pts:roi.phone,color:'#EA580C',fill:'rgba(234,88,12,0.2)'};
    ctx.beginPath(); ctx.moveTo(z.pts[0][0],z.pts[0][1]);
    for(let i=1;i<z.pts.length;i++) ctx.lineTo(z.pts[i][0],z.pts[i][1]);
    if(z.pts.length===4) ctx.closePath();
    ctx.fillStyle=z.fill; ctx.fill();
    ctx.strokeStyle=z.color; ctx.lineWidth=2; ctx.setLineDash([6,3]); ctx.stroke(); ctx.setLineDash([]);
    z.pts.forEach((p,i)=>{
      ctx.beginPath(); ctx.arc(p[0],p[1],5,0,Math.PI*2);
      ctx.fillStyle=z.color; ctx.fill();
      ctx.fillStyle='#fff'; ctx.font='bold 10px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(i+1,p[0],p[1]);
    });
    // Label
    if(z.pts.length>=1){
      ctx.fillStyle='#EA580C'; ctx.font='bold 11px Inter,sans-serif'; ctx.textAlign='left'; ctx.textBaseline='top';
      ctx.fillText('Phone Zone',z.pts[0][0]+8,z.pts[0][1]+4);
    }
  }
  // Counting line
  if(roi.line.length>0){
    ctx.beginPath(); ctx.moveTo(roi.line[0][0],roi.line[0][1]);
    if(roi.line.length>1) ctx.lineTo(roi.line[1][0],roi.line[1][1]);
    ctx.strokeStyle='#EF4444'; ctx.lineWidth=3; ctx.stroke();
    roi.line.forEach((p,i)=>{
      ctx.beginPath(); ctx.arc(p[0],p[1],6,0,Math.PI*2);
      ctx.fillStyle='#EF4444'; ctx.fill();
      ctx.fillStyle='#fff'; ctx.font='bold 10px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(i+1,p[0],p[1]);
    });
  }
}

function roiSave(){
  const sx=roi.sx, sy=roi.sy;
  const cfg={
    type:'zones',
    upper_box:roi.upper.map(p=>[Math.round(p[0]*sx),Math.round(p[1]*sy)]),
    lower_box:roi.lower.map(p=>[Math.round(p[0]*sx),Math.round(p[1]*sy)]),
    line_points:roi.line.map(p=>[Math.round(p[0]*sx),Math.round(p[1]*sy)]),
    description:'Polygon zones + tilted counting line'
  };
  if(roi.phone.length>=3)
    cfg.phone_roi=roi.phone.map(p=>[Math.round(p[0]*sx),Math.round(p[1]*sy)]);
  fetch('/api/roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
    .then(r=>r.json()).then(d=>{
      if(d.success){
        const lbl=document.getElementById('roiStepLabel');
        if(lbl) lbl.textContent='✓ Saved! Restarting stream...';
        setTimeout(()=>fetch('/start').then(()=>{
          document.getElementById('videoFeed').src='/video_feed?'+Date.now();
        }),800);
      } else alert('Failed to save: '+(d.error||'Unknown error'));
    }).catch(()=>alert('Save failed'));
}
// ─────────────────────────────────────────────────────────────────────────────

// Init reports
(function(){
  const dp=document.getElementById('reportDate');
  if(dp){
    dp.value=localToday();
    dp.addEventListener('change',()=>refreshReports());
  }
  // Initial load
  setTimeout(()=>{
    refreshReports();
    loadWeeklyReport();
  },500);
  // Auto-refresh every 10 seconds
  setInterval(refreshReports,10000);
  setInterval(loadWeeklyReport,30000);
})();
