
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

let landmarker = null;
let loadingPromise = null;

export async function initPoseAI(statusCb = () => {}) {
  if (landmarker) return landmarker;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    statusCb("AIモデルを読み込み中…");
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
      outputSegmentationMasks: false
    });

    statusCb("AIモデル準備完了");
    return landmarker;
  })();

  return loadingPromise;
}

export function detectPose(video) {
  if (!landmarker || !video || video.readyState < 2) return null;
  const ts = performance.now();
  return landmarker.detectForVideo(video, ts);
}

function visible(lm) {
  return lm && (lm.visibility == null || lm.visibility > 0.35);
}

export function drawPose(canvas, landmarks) {
  if (!canvas || !landmarks) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,rect.width,rect.height);

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#55E3BE";
  ctx.fillStyle = "#55E3BE";
  ctx.shadowColor = "rgba(0,0,0,.35)";
  ctx.shadowBlur = 3;

  for (const [a,b] of CONNECTIONS) {
    const p1 = landmarks[a], p2 = landmarks[b];
    if (!visible(p1) || !visible(p2)) continue;
    ctx.beginPath();
    ctx.moveTo(p1.x * rect.width, p1.y * rect.height);
    ctx.lineTo(p2.x * rect.width, p2.y * rect.height);
    ctx.stroke();
  }

  const keypoints = [11,12,13,14,15,16,23,24,25,26,27,28];
  for (const i of keypoints) {
    const p = landmarks[i];
    if (!visible(p)) continue;
    ctx.beginPath();
    ctx.arc(p.x * rect.width, p.y * rect.height, 5, 0, Math.PI*2);
    ctx.fill();
  }
}

function angleABC(a,b,c) {
  if (!a || !b || !c) return null;
  const v1 = {x:a.x-b.x, y:a.y-b.y};
  const v2 = {x:c.x-b.x, y:c.y-b.y};
  const dot = v1.x*v2.x + v1.y*v2.y;
  const n1 = Math.hypot(v1.x,v1.y), n2 = Math.hypot(v2.x,v2.y);
  if (!n1 || !n2) return null;
  const cos = Math.max(-1, Math.min(1, dot/(n1*n2)));
  return Math.round(Math.acos(cos) * 180/Math.PI);
}

function lineAngle(a,b) {
  if (!a || !b) return null;
  let deg = Math.atan2(b.y-a.y, b.x-a.x) * 180/Math.PI;
  deg = Math.abs(deg);
  if (deg > 90) deg = 180-deg;
  return Math.round(deg);
}

export function calculateMetrics(lm, direction="後方") {
  if (!lm || lm.length < 29) return null;

  const leftElbow = angleABC(lm[11],lm[13],lm[15]);
  const rightElbow = angleABC(lm[12],lm[14],lm[16]);
  const leftKnee = angleABC(lm[23],lm[25],lm[27]);
  const rightKnee = angleABC(lm[24],lm[26],lm[28]);

  const shoulderTilt = lineAngle(lm[11],lm[12]);
  const hipTilt = lineAngle(lm[23],lm[24]);

  const shoulderMid = {
    x:(lm[11].x+lm[12].x)/2,
    y:(lm[11].y+lm[12].y)/2
  };
  const hipMid = {
    x:(lm[23].x+lm[24].x)/2,
    y:(lm[23].y+lm[24].y)/2
  };
  const vertical = {x:hipMid.x, y:hipMid.y-0.2};
  const spineTilt = angleABC(vertical, hipMid, shoulderMid);

  return {
    direction,
    leftElbow,
    rightElbow,
    leftKnee,
    rightKnee,
    shoulderTilt,
    hipTilt,
    spineTilt
  };
}

export function buildAssessment(m) {
  if (!m) return null;

  // This is a transparent heuristic score for the current frame, not a professional diagnosis.
  let score = 100;
  const notes = [];

  const kneeVals = [m.leftKnee,m.rightKnee].filter(Number.isFinite);
  if (kneeVals.length) {
    const avgKnee = kneeVals.reduce((a,b)=>a+b,0)/kneeVals.length;
    if (avgKnee > 175) {
      score -= 8;
      notes.push("膝が伸び気味です");
    } else if (avgKnee < 115) {
      score -= 6;
      notes.push("膝の曲がりが深めです");
    } else {
      notes.push("膝の角度は安定しています");
    }
  }

  if (Number.isFinite(m.spineTilt)) {
    if (m.spineTilt < 8) {
      score -= 8;
      notes.push("上体の前傾が小さめです");
    } else if (m.spineTilt > 55) {
      score -= 8;
      notes.push("上体の前傾が大きめです");
    } else {
      notes.push("前傾角度は解析範囲内です");
    }
  }

  if (Number.isFinite(m.shoulderTilt) && Number.isFinite(m.hipTilt)) {
    const diff = Math.abs(m.shoulderTilt - m.hipTilt);
    if (diff > 25) {
      score -= 6;
      notes.push("肩と腰の傾き差が大きめです");
    }
  }

  score = Math.max(0, Math.min(100, Math.round(score)));
  return {score, notes};
}
