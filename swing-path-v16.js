
let enabled = true;
let points = [];
let phase = "address";
let lastPoint = null;
let frameCount = 0;

export function resetSwingPath(){
  points = [];
  lastPoint = null;
  phase = "address";
  frameCount = 0;
}

export function setSwingPathEnabled(v){ enabled = !!v; }
export function isSwingPathEnabled(){ return enabled; }

function mid(a,b){
  if(!a || !b) return null;
  return {x:(a.x+b.x)/2,y:(a.y+b.y)/2};
}
function dist(a,b){ return Math.hypot(a.x-b.x,a.y-b.y); }

export function updateSwingPath(lm){
  if(!enabled || !lm || lm.length < 29) return;
  const wrist = mid(lm[15],lm[16]);
  const shoulder = mid(lm[11],lm[12]);
  const hip = mid(lm[23],lm[24]);
  if(!wrist || !shoulder || !hip) return;

  frameCount++;

  // Simple phase estimator based on wrist height relative to torso.
  const torso = Math.max(0.05, dist(shoulder,hip));
  const relY = (hip.y-wrist.y)/torso;

  if(points.length < 5) phase = "address";
  else if(relY > 1.15) phase = "top";
  else if(lastPoint && wrist.y > lastPoint.y + 0.004 && points.some(p=>p.phase==="top")) phase = "downswing";
  else if(points.some(p=>p.phase==="downswing") && wrist.x !== lastPoint?.x) phase = "follow";
  else if(!points.some(p=>p.phase==="top")) phase = "backswing";

  if(!lastPoint || dist(wrist,lastPoint)>0.002){
    points.push({x:wrist.x,y:wrist.y,phase});
    if(points.length>360) points.shift();
    lastPoint=wrist;
  }
}

function videoRect(video,canvas){
  const cw=canvas.width, ch=canvas.height;
  const vw=video.videoWidth||cw, vh=video.videoHeight||ch;
  const s=Math.min(cw/vw,ch/vh);
  const w=vw*s,h=vh*s;
  return {x:(cw-w)/2,y:(ch-h)/2,w,h};
}

export function drawSwingPath(canvas,video){
  if(!enabled || points.length<2) return;
  const ctx=canvas.getContext("2d");
  const r=videoRect(video,canvas);
  const groups = [
    ["backswing","#FFD84D"],
    ["top","#FFD84D"],
    ["downswing","#FFFFFF"],
    ["follow","#55E68A"],
    ["address","#FFD84D"]
  ];

  ctx.save();
  ctx.lineCap="round";
  ctx.lineJoin="round";
  ctx.shadowColor="rgba(0,0,0,.45)";
  ctx.shadowBlur=3;

  for(const [name,color] of groups){
    let started=false;
    ctx.beginPath();
    for(const p of points){
      if(p.phase!==name){ started=false; continue; }
      const x=r.x+p.x*r.w, y=r.y+p.y*r.h;
      if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y);
    }
    ctx.strokeStyle=color;
    ctx.lineWidth=4;
    ctx.stroke();
  }
  ctx.restore();
}

export function getSwingPathSummary(){
  return {count:points.length, phase};
}
