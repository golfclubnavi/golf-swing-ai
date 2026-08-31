
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
let smoothed = null;
let previousForSpeed = null;

const SMOOTH_ALPHA = 0.12;
const LOW_CONF_ALPHA = 0.025;
const MIN_VISIBILITY = 0.62;
const DEADBAND = 0.0015;
const MAX_STEP = 0.035;

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
      minPoseDetectionConfidence: 0.58,
      minPosePresenceConfidence: 0.58,
      minTrackingConfidence: 0.65,
      outputSegmentationMasks: false
    });

    statusCb("AIモデル準備完了");
    return landmarker;
  })();

  return loadingPromise;
}

export function resetSmoothing() {
  smoothed = null;
  previousForSpeed = null;
}

function clampDelta(d) {
  if (Math.abs(d) <= DEADBAND) return 0;
  return Math.max(-MAX_STEP, Math.min(MAX_STEP, d));
}

function smoothLandmarks(lm) {
  if (!lm) return null;
  if (!smoothed || smoothed.length !== lm.length) {
    smoothed = lm.map(p => ({...p}));
    return smoothed;
  }

  smoothed = lm.map((p, i) => {
    const old = smoothed[i] || p;
    const vis = p.visibility ?? 1;
    const a = vis >= MIN_VISIBILITY ? SMOOTH_ALPHA : LOW_CONF_ALPHA;

    const dx = clampDelta(p.x - old.x);
    const dy = clampDelta(p.y - old.y);
    const dz = clampDelta((p.z ?? 0) - (old.z ?? 0));

    return {
      ...p,
      x: old.x + dx * a,
      y: old.y + dy * a,
      z: (old.z ?? 0) + dz * a,
      visibility: vis
    };
  });

  return smoothed;
}

export function detectPose(video) {
  if (!landmarker || !video || video.readyState < 2) return null;
  const timestampMs = Math.max(0, video.currentTime * 1000);
  const result = landmarker.detectForVideo(video, timestampMs);
  const raw = result?.landmarks?.[0];
  if (!raw) return { landmarks: [] };
  return { ...result, landmarks: [smoothLandmarks(raw)] };
}

function visible(lm) {
  return lm && (lm.visibility == null || lm.visibility >= MIN_VISIBILITY);
}

function getVideoContentRect(video, canvas) {
  const box = canvas.getBoundingClientRect();
  const vw = video?.videoWidth || box.width;
  const vh = video?.videoHeight || box.height;
  const scale = Math.min(box.width / vw, box.height / vh);
  const drawW = vw * scale;
  const drawH = vh * scale;

  return {
    x: (box.width - drawW) / 2,
    y: (box.height - drawH) / 2,
    width: drawW,
    height: drawH
  };
}

export function drawPose(canvas, video, landmarks) {
  if (!canvas || !video || !landmarks) return;

  const box = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(box.width * dpr));
  canvas.height = Math.max(1, Math.round(box.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,box.width,box.height);

  const r = getVideoContentRect(video, canvas);
  const mapPoint = p => [r.x + p.x * r.width, r.y + p.y * r.height];

  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#39E7C1";
  ctx.fillStyle = "#39E7C1";
  ctx.shadowColor = "rgba(0,0,0,.38)";
  ctx.shadowBlur = 2;

  for (const [a,b] of CONNECTIONS) {
    const p1 = landmarks[a], p2 = landmarks[b];
    if (!visible(p1) || !visible(p2)) continue;

    const [x1,y1] = mapPoint(p1);
    const [x2,y2] = mapPoint(p2);

    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x2,y2);
    ctx.stroke();
  }

  for (const i of KEYPOINTS) {
    const p = landmarks[i];
    if (!visible(p)) continue;

    const [x,y] = mapPoint(p);
    ctx.beginPath();
    ctx.arc(x,y,5,0,Math.PI*2);
    ctx.fill();
  }
}

function angleABC(a,b,c) {
  if (!visible(a) || !visible(b) || !visible(c)) return null;
  const v1 = {x:a.x-b.x, y:a.y-b.y};
  const v2 = {x:c.x-b.x, y:c.y-b.y};
  const dot = v1.x*v2.x + v1.y*v2.y;
  const n1 = Math.hypot(v1.x,v1.y), n2 = Math.hypot(v2.x,v2.y);
  if (!n1 || !n2) return null;
  const cos = Math.max(-1, Math.min(1, dot/(n1*n2)));
  return Math.round(Math.acos(cos) * 180/Math.PI);
}

function signedLineAngle(a,b) {
  if (!visible(a) || !visible(b)) return null;
  return Math.atan2(b.y-a.y, b.x-a.x) * 180/Math.PI;
}

function displayLineAngle(a,b) {
  const raw = signedLineAngle(a,b);
  if (!Number.isFinite(raw)) return null;
  let deg = Math.abs(raw);
  if (deg > 90) deg = 180-deg;
  return Math.round(deg);
}

export function calculateMetrics(lm, direction="後方") {
  if (!lm || lm.length < 29) return null;

  const leftElbow = angleABC(lm[11],lm[13],lm[15]);
  const rightElbow = angleABC(lm[12],lm[14],lm[16]);
  const leftKnee = angleABC(lm[23],lm[25],lm[27]);
  const rightKnee = angleABC(lm[24],lm[26],lm[28]);

  const shoulderTilt = displayLineAngle(lm[11],lm[12]);
  const hipTilt = displayLineAngle(lm[23],lm[24]);

  let spineTilt = null;
  if (visible(lm[11]) && visible(lm[12]) && visible(lm[23]) && visible(lm[24])) {
    const shoulderMid = {
      x:(lm[11].x+lm[12].x)/2,
      y:(lm[11].y+lm[12].y)/2,
      visibility:1
    };
    const hipMid = {
      x:(lm[23].x+lm[24].x)/2,
      y:(lm[23].y+lm[24].y)/2,
      visibility:1
    };
    const vertical = {x:hipMid.x, y:hipMid.y-0.2, visibility:1};
    spineTilt = angleABC(vertical, hipMid, shoulderMid);
  }

  return {
    direction,
    leftElbow,rightElbow,leftKnee,rightKnee,
    shoulderTilt,hipTilt,spineTilt,
    shoulderRaw: signedLineAngle(lm[11],lm[12]),
    hipRaw: signedLineAngle(lm[23],lm[24])
  };
}

export function calculateSpeedMetrics(lm, currentTimeSec) {
  if (!lm || !Number.isFinite(currentTimeSec)) return null;

  const shoulderWidth = (
    visible(lm[11]) && visible(lm[12])
      ? Math.hypot(lm[12].x-lm[11].x, lm[12].y-lm[11].y)
      : null
  );

  const leftWrist = visible(lm[15]) ? lm[15] : null;
  const rightWrist = visible(lm[16]) ? lm[16] : null;

  let wristMid = null;
  if (leftWrist && rightWrist) {
    wristMid = {
      x:(leftWrist.x+rightWrist.x)/2,
      y:(leftWrist.y+rightWrist.y)/2
    };
  } else if (leftWrist) {
    wristMid = {x:leftWrist.x,y:leftWrist.y};
  } else if (rightWrist) {
    wristMid = {x:rightWrist.x,y:rightWrist.y};
  }

  const shoulderAngle = signedLineAngle(lm[11],lm[12]);
  const hipAngle = signedLineAngle(lm[23],lm[24]);

  const result = {
    wristSpeedBodyPerSec:null,
    shoulderLineDegPerSec:null,
    hipLineDegPerSec:null
  };

  if (previousForSpeed && currentTimeSec > previousForSpeed.t) {
    const dt = currentTimeSec - previousForSpeed.t;

    if (wristMid && previousForSpeed.wristMid && shoulderWidth && shoulderWidth > 0.01) {
      const dist = Math.hypot(
        wristMid.x-previousForSpeed.wristMid.x,
        wristMid.y-previousForSpeed.wristMid.y
      );
      result.wristSpeedBodyPerSec = dist / shoulderWidth / dt;
    }

    const angularDelta = (a,b) => {
      if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
      let d = a-b;
      while (d > 180) d -= 360;
      while (d < -180) d += 360;
      return Math.abs(d) / dt;
    };

    result.shoulderLineDegPerSec = angularDelta(shoulderAngle, previousForSpeed.shoulderAngle);
    result.hipLineDegPerSec = angularDelta(hipAngle, previousForSpeed.hipAngle);
  }

  previousForSpeed = {
    t:currentTimeSec,
    wristMid,
    shoulderAngle,
    hipAngle
  };

  return result;
}

export function buildAssessment(m) {
  if (!m) return null;

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

  return {
    score: Math.max(0,Math.min(100,Math.round(score))),
    notes
  };
}
