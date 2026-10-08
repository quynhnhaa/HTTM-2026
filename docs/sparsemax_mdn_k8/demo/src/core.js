// ─── Lõi tính toán: LSTM + MDN head, chạy hoàn toàn trong trình duyệt ───
// Không phụ thuộc DOM. Cùng một file được chạy thử bằng node và nhúng vào trang.

const DT = 0.1, HORIZON = 48, NSTEP = 32, HID = 8;

const sig = x => 1 / (1 + Math.exp(-x));

// LSTM(4→8) một lớp, lấy bước cuối, rồi Linear(8→48·6K).
// Thứ tự cổng của PyTorch: i, f, g, o.
function forward(m, X) {
  const K = m.K, H = HID;
  const h = new Float64Array(H), c = new Float64Array(H), g = new Float64Array(4 * H);
  for (let t = 0; t < NSTEP; t++) {
    for (let r = 0; r < 4 * H; r++) {
      let s = m.bi[r] + m.bh[r];
      for (let j = 0; j < 4; j++) s += m.Wi[r * 4 + j] * X[t * 4 + j];
      for (let j = 0; j < H; j++) s += m.Wh[r * H + j] * h[j];
      g[r] = s;
    }
    for (let j = 0; j < H; j++) {
      const i = sig(g[j]), f = sig(g[H + j]), gg = Math.tanh(g[2 * H + j]), o = sig(g[3 * H + j]);
      c[j] = f * c[j] + i * gg;
      h[j] = o * Math.tanh(c[j]);
    }
  }
  const O = 6 * K, out = new Float64Array(HORIZON * O);
  for (let r = 0; r < HORIZON * O; r++) {
    let s = m.bo[r];
    for (let j = 0; j < H; j++) s += m.Wo[r * H + j] * h[j];
    out[r] = s;
  }
  return out; // row-major (48, 6K)
}

function softmax(z) {
  const mx = Math.max(...z);
  const e = z.map(v => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

// Sparsemax (Martins & Astudillo 2016): phép chiếu Euclid lên simplex.
// Khác softmax ở chỗ cho ra số 0 THẬT SỰ.
function sparsemax(z) {
  const n = z.length;
  const mx = Math.max(...z);
  const y = z.map(v => v - mx);                 // bất biến theo dịch chuyển
  const s = [...y].sort((a, b) => b - a);
  const cs = []; let cum = 0;
  for (let i = 0; i < n; i++) { cum += s[i]; cs.push(cum); }
  let k = 1;
  for (let i = 0; i < n; i++) if (1 + (i + 1) * s[i] > cs[i]) k = i + 1;
  const tau = (cs[k - 1] - 1) / k;
  return y.map(v => Math.max(v - tau, 0));
}

// Cắt 6K kênh thô của một mốc thời gian thành tham số GMM.
function decode(out, K, t, norm) {
  const b = t * 6 * K, comp = [];
  const logit = [];
  for (let k = 0; k < K; k++) logit.push(out[b + 5 * K + k]);
  const pi = norm === 'sparsemax' ? sparsemax(logit) : softmax(logit);
  for (let k = 0; k < K; k++) {
    comp.push({
      pi: pi[k],
      mx: out[b + k],
      my: out[b + K + k],
      sx: Math.exp(out[b + 2 * K + k]),
      sy: Math.exp(out[b + 3 * K + k]),
      rho: Math.tanh(out[b + 4 * K + k]),
    });
  }
  return comp;
}

// Elip đường mức: bán trục THẬT là căn trị riêng của Σ, không phải σ.
// r = bán kính chuẩn hoá (1.510 cho 68 %, 2.448 cho 95 %, 1.0 cho "1σ").
function ellipse(c, r) {
  const a = c.sx * c.sx, b = c.rho * c.sx * c.sy, d = c.sy * c.sy;
  const tr = a + d, det = a * d - b * b;
  const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  const ang = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  return { rx: Math.sqrt(Math.max(l1, 0)) * r, ry: Math.sqrt(Math.max(l2, 0)) * r, ang };
}

// ─── Tiền xử lý: đúng quy ước đã kiểm trên dữ liệu thật ───

// Lấy n điểm cách đều `d` mét dọc đường đã vẽ, ĐI NGƯỢC từ điểm cuối.
// Đường ngắn quá thì ngoại suy thẳng về phía sau.
function resampleBack(pts, n, d) {
  if (pts.length === 0) return null;
  if (pts.length === 1) {
    const [x, y] = pts[0];
    return Array.from({ length: n }, () => [x, y]);
  }
  const out = new Array(n);
  out[n - 1] = pts[pts.length - 1].slice();
  let i = pts.length - 1, cur = out[n - 1].slice(), rem = d;
  let lastDir = null;
  for (let k = n - 2; k >= 0; k--) {
    while (i > 0) {
      const prev = pts[i - 1];
      const seg = Math.hypot(cur[0] - prev[0], cur[1] - prev[1]);
      if (seg >= rem && seg > 1e-12) {
        const t = rem / seg;
        lastDir = [(prev[0] - cur[0]) / seg, (prev[1] - cur[1]) / seg];
        cur = [cur[0] + (prev[0] - cur[0]) * t, cur[1] + (prev[1] - cur[1]) * t];
        rem = d;
        break;
      }
      rem -= seg;
      if (seg > 1e-12) lastDir = [(prev[0] - cur[0]) / seg, (prev[1] - cur[1]) / seg];
      cur = prev.slice();
      i--;
    }
    if (i === 0) {                       // hết đường: ngoại suy ngược
      const dir = lastDir || [0, -1];
      cur = [cur[0] + dir[0] * rem, cur[1] + dir[1] * rem];
      rem = d;
    }
    out[k] = cur.slice();
  }
  return out;
}

// Chuyển sang hệ ego: gốc = bước 32, hướng đi → trục +y.
// φ lấy từ dịch chuyển 24 bước cuối (khớp dữ liệu gốc nhất: lệch trung vị 0,54°).
function toEgo(P) {
  const n = P.length, o = P[n - 1];
  const pick = i => P[Math.max(0, n - 1 - i)];
  let dx = o[0] - pick(24)[0], dy = o[1] - pick(24)[1];
  if (Math.hypot(dx, dy) < 0.05) { dx = o[0] - P[0][0]; dy = o[1] - P[0][1]; }
  if (Math.hypot(dx, dy) < 1e-9) { dx = 0; dy = 1; }
  const phi = Math.atan2(dy, dx) - Math.PI / 2;
  const c = Math.cos(phi), s = Math.sin(phi);
  // R(φ) = [[cos, sin], [−sin, cos]] — đúng build_R trong helper.py
  const ego = P.map(p => {
    const ux = p[0] - o[0], uy = p[1] - o[1];
    return [c * ux + s * uy, -s * ux + c * uy];
  });
  return { ego, phi, origin: o };
}

// Chiều ngược: ego → khung người dùng đã vẽ. R(φ)⁻¹ = quay +φ.
function toWorld(p, phi, origin) {
  const c = Math.cos(phi), s = Math.sin(phi);
  return [c * p[0] - s * p[1] + origin[0], s * p[0] + c * p[1] + origin[1]];
}

// Ghép (32, 4): hai cột vị trí + hai cột vận tốc.
// v[t] = (p[t] − p[t−1]) × 10 ; v[0] sao chép từ v[1] — đúng như file gốc.
function buildInput(ego) {
  const X = new Float64Array(NSTEP * 4);
  for (let t = 0; t < NSTEP; t++) { X[t * 4] = ego[t][0]; X[t * 4 + 1] = ego[t][1]; }
  for (let t = 1; t < NSTEP; t++) {
    X[t * 4 + 2] = (ego[t][0] - ego[t - 1][0]) / DT;
    X[t * 4 + 3] = (ego[t][1] - ego[t - 1][1]) / DT;
  }
  X[2] = X[6]; X[3] = X[7];
  return X;
}

if (typeof module !== 'undefined') {
  module.exports = { forward, decode, softmax, sparsemax, ellipse, resampleBack, toEgo, toWorld, buildInput, DT, HORIZON, NSTEP };
}
