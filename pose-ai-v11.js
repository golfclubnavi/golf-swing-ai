
import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm";

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
  [27,29],[29,31],[28,30],[30,32]
];

const KEYPOINTS = [11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32];

let landmarker = null;
let loadingPromise = null;
let smoothState = null;
let history = null;
let previousForSpeed = null;

const MIN_VISIBILITY = 0.62;
const HISTORY_SIZE = 5;

export async function initPoseAI(statusCb = () => {}) {
  if (landmarker) return landmarker;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    statusCb("AIモデルを読み込み中…");
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.60,
      minPosePresenceConfidence: 0.60,
      minTrackingConfidence: 0.68,
      outputSegmentationMasks: false
    });

    statusCb("AIモデル準備完了");
    return landmarker;
  })();

  return loadingPromise;
}

export function resetSmoothing() {
  smoothState = null;
  history = null;
  previousForSpeed = null;
}

function median(values) {
  const a = values.filter(Number.isFinite).sort((x,y)=>x-y);
  if (!a.length) return null;
  const m = Math.floor(a.length/2);
  return a.length % 2 ? a[m] : (a[m-1]+a[m])/2;
}

function robustSmooth(lm) {
  if (!lm) return null;

  if (!history || history.length !== lm.length) {
    history = lm.map(()=>[]);
    smoothState = lm.map(p=>({...p}));
  }

  return lm.map((p,i)=>{
    const h = history[i];
    h.push({x:p.x,y:p.y,z:p.z??0,visibility:p.visibility??1});
    if (h.length > HISTORY_SIZE) h.shift();

    const medX = median(h.map(v=>v.x)) ?? p.x;
    const medY = median(h.map(v=>v.y)) ?? p.y;
    const medZ = median(h.map(v=>v.z)) ?? (p.z??0);

    const prev = smoothState[i] || p;
    const rawMove = Math.hypot(medX-prev.x, medY-prev.y);

    // More smoothing when nearly still, more responsiveness during a real swing.
    let alpha;
    if ((p.visibility ?? 1) < MIN_VISIBILITY) alpha = 0.03;
    else if (rawMove < 0.004) alpha = 0.10;
    else if (rawMove < 0.012) alpha = 0.18;
    else alpha = 0.34;

    const maxStep = 0.045;
    const dx = Math.max(-maxStep,Math.min(maxStep,medX-prev.x));
    const dy = Math.max(-maxStep,Math.min(maxStep,medY-prev.y));
    const dz = Math.max(-maxStep,Math.min(maxStep,medZ-(prev.z??0)));

    const next = {
      ...p,
      x: prev.x + dx*alpha,
      y: prev.y + dy*alpha,
      z: (prev.z??0) + dz*alpha,
      visibility: p.visibility??1
    };

    smoothState[i] = next;
    return next;
  });
}

export function detectPose(video) {
  if (!landmarker || !video || video.readyState < 2) return null;
  const timestampMs = performance.now();
  const result = landmarker.detectForVideo(video, timestampMs);
  const raw = result?.landmarks?.[0];
  if (!raw) return {landmarks:[]};
  return {...result, landmarks:[robustSmooth(raw)]};
}

function visible(p) {
  return p && (p.visibility == null || p.visibility >= MIN_VISIBILITY);
}

function getVideoContentRect(video, canvas) {
  const box = canvas.getBoundingClientRect();
  const vw = video?.videoWidth || box.width;
  const vh = video?.videoHeight || box.height;
  const scale = Math.min(box.width/vw, box.height/vh);
  const w = vw*scale, h = vh*scale;
  return {x:(box.width-w)/2,y:(box.height-h)/2,width:w,height:h};
}

export function drawPose(canvas, video, lm) {
  if (!canvas || !video || !lm) return;
  const box = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width*dpr);
  canvas.height = Math.round(box.height*dpr);

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,box.width,box.height);

  const r = getVideoContentRect(video,canvas);
  const pt = p=>[r.x+p.x*r.width,r.y+p.y*r.height];

  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#39E7C1";
  ctx.fillStyle = "#39E7C1";

  for (const [a,b] of CONNECTIONS) {
    if (!visible(lm[a]) || !visible(lm[b])) continue;
    const [x1,y1] = pt(lm[a]), [x2,y2] = pt(lm[b]);
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }

  for (const i of KEYPOINTS) {
    if (!visible(lm[i])) continue;
    const [x,y] = pt(lm[i]);
    ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fill();
  }
}

function angleABC(a,b,c) {
  if (!visible(a)||!visible(b)||!visible(c)) return null;
  const v1={x:a.x-b.x,y:a.y-b.y}, v2={x:c.x-b.x,y:c.y-b.y};
  const dot=v1.x*v2.x+v1.y*v2.y, n1=Math.hypot(v1.x,v1.y), n2=Math.hypot(v2.x,v2.y);
  if(!n1||!n2) return null;
  const cos=Math.max(-1,Math.min(1,dot/(n1*n2)));
  return Math.round(Math.acos(cos)*180/Math.PI);
}

function signedLineAngle(a,b) {
  if(!visible(a)||!visible(b)) return null;
  return Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI;
}

function displayLineAngle(a,b){
  const raw=signedLineAngle(a,b);
  if(!Number.isFinite(raw)) return null;
  let d=Math.abs(raw);
  if(d>90)d=180-d;
  return Math.round(d);
}

export function calculateMetrics(lm,direction="後方"){
  if(!lm||lm.length<29)return null;
  const leftElbow=angleABC(lm[11],lm[13],lm[15]);
  const rightElbow=angleABC(lm[12],lm[14],lm[16]);
  const leftKnee=angleABC(lm[23],lm[25],lm[27]);
  const rightKnee=angleABC(lm[24],lm[26],lm[28]);
  const shoulderTilt=displayLineAngle(lm[11],lm[12]);
  const hipTilt=displayLineAngle(lm[23],lm[24]);

  let spineTilt=null;
  if(visible(lm[11])&&visible(lm[12])&&visible(lm[23])&&visible(lm[24])){
    const sm={x:(lm[11].x+lm[12].x)/2,y:(lm[11].y+lm[12].y)/2,visibility:1};
    const hm={x:(lm[23].x+lm[24].x)/2,y:(lm[23].y+lm[24].y)/2,visibility:1};
    spineTilt=angleABC({x:hm.x,y:hm.y-.2,visibility:1},hm,sm);
  }

  return {
    direction,leftElbow,rightElbow,leftKnee,rightKnee,
    shoulderTilt,hipTilt,spineTilt,
    shoulderRaw:signedLineAngle(lm[11],lm[12]),
    hipRaw:signedLineAngle(lm[23],lm[24])
  };
}

export function calculateSpeedMetrics(lm,t){
  if(!lm||!Number.isFinite(t))return null;

  const sw=visible(lm[11])&&visible(lm[12])
    ? Math.hypot(lm[12].x-lm[11].x,lm[12].y-lm[11].y)
    : null;

  let wristMid=null;
  const lw=visible(lm[15])?lm[15]:null, rw=visible(lm[16])?lm[16]:null;
  if(lw&&rw) wristMid={x:(lw.x+rw.x)/2,y:(lw.y+rw.y)/2};
  else if(lw) wristMid={x:lw.x,y:lw.y};
  else if(rw) wristMid={x:rw.x,y:rw.y};

  const shoulder=signedLineAngle(lm[11],lm[12]);
  const hip=signedLineAngle(lm[23],lm[24]);

  const out={wristSpeedBodyPerSec:null,shoulderLineDegPerSec:null,hipLineDegPerSec:null};

  if(previousForSpeed && t>previousForSpeed.t){
    const dt=t-previousForSpeed.t;
    if(dt>0.002 && dt<0.5){
      if(wristMid&&previousForSpeed.wristMid&&sw&&sw>.01){
        const d=Math.hypot(wristMid.x-previousForSpeed.wristMid.x,wristMid.y-previousForSpeed.wristMid.y);
        out.wristSpeedBodyPerSec=d/sw/dt;
      }
      const av=(a,b)=>{
        if(!Number.isFinite(a)||!Number.isFinite(b))return null;
        let d=a-b; while(d>180)d-=360; while(d<-180)d+=360;
        return Math.abs(d)/dt;
      };
      out.shoulderLineDegPerSec=av(shoulder,previousForSpeed.shoulder);
      out.hipLineDegPerSec=av(hip,previousForSpeed.hip);
    }
  }

  previousForSpeed={t,wristMid,shoulder,hip};
  return out;
}

export function buildAssessment(m){
  if(!m)return null;
  let score=100; const notes=[];
  const knees=[m.leftKnee,m.rightKnee].filter(Number.isFinite);
  if(knees.length){
    const avg=knees.reduce((a,b)=>a+b,0)/knees.length;
    if(avg>175){score-=8;notes.push("膝が伸び気味です");}
    else if(avg<115){score-=6;notes.push("膝の曲がりが深めです");}
    else notes.push("膝の角度は安定しています");
  }
  if(Number.isFinite(m.spineTilt)){
    if(m.spineTilt<8){score-=8;notes.push("上体の前傾が小さめです");}
    else if(m.spineTilt>55){score-=8;notes.push("上体の前傾が大きめです");}
    else notes.push("前傾角度は解析範囲内です");
  }
  if(Number.isFinite(m.shoulderTilt)&&Number.isFinite(m.hipTilt)){
    if(Math.abs(m.shoulderTilt-m.hipTilt)>25){score-=6;notes.push("肩と腰の傾き差が大きめです");}
  }
  return {score:Math.max(0,Math.min(100,Math.round(score))),notes};
}
