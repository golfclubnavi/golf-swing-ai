
const snapshots = {address:null,top:null,impact:null};

function mid(a,b){return {x:(a.x+b.x)/2,y:(a.y+b.y)/2};}
function deg(a,b){
  let d=Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI;
  while(d>90)d-=180; while(d<-90)d+=180;
  return d;
}
function joint(a,b,c){
  const ab={x:a.x-b.x,y:a.y-b.y}, cb={x:c.x-b.x,y:c.y-b.y};
  const dot=ab.x*cb.x+ab.y*cb.y;
  const m=Math.hypot(ab.x,ab.y)*Math.hypot(cb.x,cb.y);
  if(!m)return null;
  return Math.acos(Math.max(-1,Math.min(1,dot/m)))*180/Math.PI;
}
function metrics(lm){
  const sh=mid(lm[11],lm[12]), hip=mid(lm[23],lm[24]);
  const wrist=mid(lm[15],lm[16]);
  return {
    shoulderTilt:Math.round(deg(lm[11],lm[12])),
    hipTilt:Math.round(deg(lm[23],lm[24])),
    spineTilt:Math.round(Math.abs(deg(hip,sh)-90)),
    leftKnee:Math.round(joint(lm[23],lm[25],lm[27])||0),
    rightKnee:Math.round(joint(lm[24],lm[26],lm[28])||0),
    wristX:+wrist.x.toFixed(3),
    wristY:+wrist.y.toFixed(3),
    headX:+((lm[7].x+lm[8].x)/2).toFixed(3)
  };
}
function scoreRange(v,min,max,soft=10){
  if(v>=min&&v<=max)return 100;
  const d=v<min?min-v:v-max;
  return Math.max(45,Math.round(100-d*(55/soft)));
}
function evaluate(name,m){
  let scores=[], notes=[];
  if(name==="address"){
    scores=[scoreRange(m.spineTilt,20,50,20),scoreRange(m.leftKnee,135,175,25),scoreRange(m.rightKnee,135,175,25)];
    if(scores[0]<80)notes.push("前傾姿勢を確認");
    if(Math.min(scores[1],scores[2])<80)notes.push("膝の曲げ量を確認");
  }else if(name==="top"){
    scores=[scoreRange(Math.abs(m.shoulderTilt),5,35,20),scoreRange(Math.abs(m.hipTilt),0,25,20)];
    if(scores[0]<80)notes.push("トップで肩の傾きを確認");
    if(scores[1]<80)notes.push("腰の傾きを確認");
  }else{
    scores=[scoreRange(m.spineTilt,15,50,25),scoreRange(Math.abs(m.shoulderTilt),0,35,25)];
    if(scores[0]<80)notes.push("インパクト時の前傾を確認");
    if(scores[1]<80)notes.push("肩のラインを確認");
  }
  const score=Math.round(scores.reduce((a,b)=>a+b,0)/scores.length);
  return {score,notes:notes.length?notes:["大きな崩れは検出されませんでした"]};
}
export function captureCheckpoint(name,lm){
  if(!lm)return null;
  const m=metrics(lm), e=evaluate(name,m);
  snapshots[name]={metrics:m,...e};
  return snapshots[name];
}
export function getCheckpoints(){return JSON.parse(JSON.stringify(snapshots));}
export function resetCheckpoints(){snapshots.address=snapshots.top=snapshots.impact=null;}
