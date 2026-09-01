
let prevGray = null;
let trackPoint = null;
let trail = [];
let enabled = false;
let lastTime = -1;

const work = document.createElement("canvas");
const ctx = work.getContext("2d", {willReadFrequently:true});

export function setBallTrackingEnabled(v){
  enabled = !!v;
  if(!enabled){
    prevGray = null;
    trackPoint = null;
    trail = [];
    lastTime = -1;
  }
}

export function isBallTrackingEnabled(){
  return enabled;
}

export function setBallStartNormalized(x,y){
  trackPoint = {x,y};
  trail = [{x,y}];
  prevGray = null;
}

export function clearBallTrail(){
  trackPoint = null;
  trail = [];
  prevGray = null;
  lastTime = -1;
}

function getVideoRect(video, overlay){
  const box = overlay.getBoundingClientRect();
  const vw = video.videoWidth || box.width;
  const vh = video.videoHeight || box.height;
  const scale = Math.min(box.width/vw, box.height/vh);
  const w = vw*scale, h = vh*scale;
  return {
    x:(box.width-w)/2,
    y:(box.height-h)/2,
    width:w,
    height:h
  };
}

function grayImage(data){
  const out = new Uint8Array(data.length/4);
  for(let i=0,j=0;i<data.length;i+=4,j++){
    out[j] = Math.round(data[i]*0.299 + data[i+1]*0.587 + data[i+2]*0.114);
  }
  return out;
}

export function processBallFrame(video, overlay){
  if(!enabled || !trackPoint || !video || video.readyState<2) return;

  const vw = video.videoWidth, vh = video.videoHeight;
  if(!vw || !vh) return;

  const maxSide = 480;
  const scale = Math.min(1, maxSide/Math.max(vw,vh));
  const w = Math.max(1,Math.round(vw*scale));
  const h = Math.max(1,Math.round(vh*scale));

  if(work.width!==w || work.height!==h){
    work.width=w; work.height=h;
    prevGray=null;
  }

  ctx.drawImage(video,0,0,w,h);
  const img = ctx.getImageData(0,0,w,h);
  const gray = grayImage(img.data);

  if(!prevGray){
    prevGray = gray;
    lastTime = video.currentTime||0;
    return;
  }

  const px = Math.round(trackPoint.x*w);
  const py = Math.round(trackPoint.y*h);

  // Search locally around last known point.
  const radius = Math.max(18, Math.round(Math.min(w,h)*0.10));
  const x0 = Math.max(1,px-radius), x1 = Math.min(w-2,px+radius);
  const y0 = Math.max(1,py-radius), y1 = Math.min(h-2,py+radius);

  let best = null;
  for(let y=y0;y<=y1;y+=2){
    for(let x=x0;x<=x1;x+=2){
      const i=y*w+x;
      const diff=Math.abs(gray[i]-prevGray[i]);

      // Prefer moving + bright pixels. This is experimental and works best
      // when the ball is clearly visible against darker background.
      const lum=gray[i];
      const score=diff*1.35 + lum*0.28;

      if(diff>18 && lum>90 && (!best || score>best.score)){
        best={x,y,score};
      }
    }
  }

  if(best){
    const nx=best.x/w, ny=best.y/h;
    const jump=Math.hypot(nx-trackPoint.x,ny-trackPoint.y);

    // reject implausibly large one-frame jumps
    if(jump < 0.18){
      trackPoint={x:nx,y:ny};
      trail.push(trackPoint);
      if(trail.length>90) trail.shift();
    }
  }

  prevGray=gray;
  lastTime=video.currentTime||0;
}

export function drawBallTrail(overlay,video){
  if(!overlay || !video) return;
  const r=getVideoRect(video,overlay);
  const c=overlay.getContext("2d");

  // Draw on top of pose overlay without clearing it.
  c.save();
  c.lineCap="round";
  c.lineJoin="round";

  if(trail.length>1){
    c.beginPath();
    trail.forEach((p,i)=>{
      const x=r.x+p.x*r.width, y=r.y+p.y*r.height;
      if(i===0)c.moveTo(x,y); else c.lineTo(x,y);
    });
    c.lineWidth=4;
    c.strokeStyle="#FFD84D";
    c.shadowColor="rgba(0,0,0,.5)";
    c.shadowBlur=3;
    c.stroke();
  }

  if(trackPoint){
    const x=r.x+trackPoint.x*r.width, y=r.y+trackPoint.y*r.height;
    c.beginPath();
    c.arc(x,y,7,0,Math.PI*2);
    c.fillStyle="#FFF35A";
    c.fill();
    c.lineWidth=2;
    c.strokeStyle="#111";
    c.stroke();
  }

  c.restore();
}

export function overlayTapToNormalized(ev, overlay, video){
  const rect=overlay.getBoundingClientRect();
  const r=getVideoRect(video,overlay);
  const x=ev.clientX-rect.left, y=ev.clientY-rect.top;
  if(x<r.x || y<r.y || x>r.x+r.width || y>r.y+r.height) return null;
  return {
    x:(x-r.x)/r.width,
    y:(y-r.y)/r.height
  };
}
