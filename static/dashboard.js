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
  const dateStr=dp?dp.value:new Date().toISOString().split('T')[0];
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
  if(!checkXLSX())return;
  fetch('/api/reports/week').then(r=>r.json()).then(d=>{
    if(!d.days||d.days.length===0){alert('No weekly data available');return;}
    const wb=XLSX.utils.book_new();
    const rows=[['Date','Day','Total IN','Total OUT','Net Inside','Violations']];
    let sumIn=0,sumOut=0,sumViol=0;
    d.days.forEach(day=>{
      const net=Math.max(day.total_in-day.total_out,0);
      rows.push([day.date, day.day_name, day.total_in, day.total_out, net, day.total_violations]);
      sumIn+=day.total_in; sumOut+=day.total_out; sumViol+=day.total_violations;
    });
    rows.push([]);
    rows.push(['Total','',sumIn,sumOut,Math.max(sumIn-sumOut,0),sumViol]);
    const ws=XLSX.utils.aoa_to_sheet(rows);
    ws['!cols']=[{wch:12},{wch:6},{wch:12},{wch:12},{wch:12},{wch:12}];
    XLSX.utils.book_append_sheet(wb, ws, 'Weekly Report');
    saveExcel(wb, `Weekly_Report_${new Date().toISOString().split('T')[0]}.xlsx`);
  }).catch(()=>alert('Failed to fetch weekly data'));
}

function downloadViolationsCSV(){
  if(!checkXLSX())return;
  fetch('/violations').then(r=>r.json()).then(d=>{
    if(!d.violations||d.violations.length===0){alert('No violations recorded');return;}
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
    saveExcel(wb, `Violations_Report_${new Date().toISOString().split('T')[0]}.xlsx`);
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
      const rv=document.getElementById('rptViolations');if(rv)rv.textContent=d.summary.total_violations;
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

function refreshReports(){
  const dp=document.getElementById('reportDate');
  const dateStr=dp?dp.value:new Date().toISOString().split('T')[0];
  loadDailyReport(dateStr);

  // Also update live stats for today
  const today=new Date().toISOString().split('T')[0];
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

  // Fetch violations for table
  fetch('/violations').then(r=>r.json()).then(d=>{
    const total=d.total||0;
    const rc=document.getElementById('rptViolCount');if(rc)rc.textContent=total+' violations recorded';
    const body=document.getElementById('rptViolBody');
    if(body){
      if(d.violations&&d.violations.length>0){
        body.innerHTML=d.violations.map((v,i)=>
          `<tr>
            <td style="padding:8px 12px;border-bottom:1px solid #F1F5F9;color:#6B7280">${i+1}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #F1F5F9;color:#1E293B;font-weight:500">${v.timestamp||'--'}</td>
            <td style="padding:8px 12px;text-align:center;border-bottom:1px solid #F1F5F9;color:#6B7280">${v.frame_number||'--'}</td>
            <td style="padding:8px 12px;text-align:center;border-bottom:1px solid #F1F5F9">${v.filename?`<img src="/violations/${v.filename}" style="width:40px;height:30px;object-fit:cover;border-radius:4px;border:1px solid #E5E7EB;cursor:pointer" onclick="window.open('/violations/${v.filename}','_blank')">`:'-'}</td>
          </tr>`
        ).join('');
      } else {
        body.innerHTML='<tr><td colspan="4" style="padding:20px;text-align:center;color:#9CA3AF">No violations recorded yet</td></tr>';
      }
    }
  }).catch(()=>{});
}

// Init reports
(function(){
  const dp=document.getElementById('reportDate');
  if(dp){
    dp.value=new Date().toISOString().split('T')[0];
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
