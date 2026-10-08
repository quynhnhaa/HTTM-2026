// Đối chiếu forward pass viết bằng JavaScript với file dự đoán mà repo đã lưu.
// Đây là thứ làm demo đáng tin: nếu số không khớp thì demo đang nói dối.
//
//     node docs/sparsemax_mdn_k8/demo/tools/verify.js
//
// Cần hai file do tools/make_presets.py sinh ra:
//   assets/_ref_softmax.json   (từ base_mdn M=3)
//   assets/_ref_sparsemax.json (từ run sparsemax trên HuggingFace)
// Thiếu file nào thì phần đó được bỏ qua.

const fs = require('fs');
const path = require('path');

const ROOT = path.dirname(__dirname);
const C = require(path.join(ROOT, 'src', 'core.js'));
const W = JSON.parse(fs.readFileSync(path.join(ROOT, 'assets', 'weights.json'), 'utf8'));

function flat(X) {                       // [32][4] -> Float64Array(128)
  const out = new Float64Array(32 * 4);
  for (let t = 0; t < 32; t++) for (let j = 0; j < 4; j++) out[t * 4 + j] = X[t][j];
  return out;
}

function check(label, refFile, modelKey, K, norm) {
  const p = path.join(ROOT, 'assets', refFile);
  if (!fs.existsSync(p)) { console.log(`-- bỏ qua ${label}: thiếu ${refFile}`); return true; }
  const R = JSON.parse(fs.readFileSync(p, 'utf8'));
  const m = W.models[modelKey];
  let dPi = 0, dMu = 0, dSig = 0, zeroJs = 0, zeroRef = 0, nk = 0, nkRef = 0;
  for (let b = 0; b < R.X.length; b++) {
    const out = C.forward(m, flat(R.X[b]));
    for (let t = 0; t < 48; t++) {
      const comp = C.decode(out, K, t, norm);
      for (let k = 0; k < K; k++) {
        dPi = Math.max(dPi, Math.abs(comp[k].pi - R.pi[b][t][k]));
        dMu = Math.max(dMu, Math.abs(comp[k].mx - R.mu[b][t][k][0]),
                            Math.abs(comp[k].my - R.mu[b][t][k][1]));
        dSig = Math.max(dSig, Math.abs(comp[k].sx - R.sigma[b][t][k][0]),
                              Math.abs(comp[k].sy - R.sigma[b][t][k][1]));
        if (comp[k].pi === 0) zeroJs++;
        if (R.pi[b][t][k] === 0) zeroRef++;
      }
      nk += comp.filter(c => c.pi > 0).length;
      nkRef += R.pi[b][t].filter(v => v > 0).length;
    }
  }
  console.log(`${label}`);
  console.log(`   mu     lệch tối đa ${(dMu * 1000).toFixed(1)} mm`);
  console.log(`   sigma  lệch tối đa ${(dSig * 1000).toFixed(1)} mm`);
  console.log(`   pi     lệch tối đa ${dPi.toExponential(2)}`);
  if (norm === 'sparsemax') {
    console.log(`   số 0 thật: JS ${zeroJs} · repo ${zeroRef}` +
                (zeroJs === zeroRef ? '  ✓ khớp' : '  ✗ LỆCH'));
    console.log(`   K trung bình: JS ${(nk / (R.X.length * 48)).toFixed(2)}` +
                ` · repo ${(nkRef / (R.X.length * 48)).toFixed(2)}`);
  }
  // Ngưỡng 25 mm: repo chạy GPU ở độ chính xác TF32 (~1e-3 tương đối),
  // bản JS dùng float64 nên thực ra chính xác hơn. Chênh này là của repo.
  const ok = dMu < 0.025 && dSig < 0.025 && (norm !== 'sparsemax' || zeroJs === zeroRef);
  console.log(ok ? '   ✓ ĐẠT\n' : '   ✗ KHÔNG ĐẠT\n');
  return ok;
}

let ok = true;
ok = check('base M=3 (softmax)', '_ref_softmax.json', '3', 3, 'softmax') && ok;
if (W.models.sp) {
  ok = check('sparsemax K≤8 (đã huấn luyện)', '_ref_sparsemax.json', 'sp', 8, 'sparsemax') && ok;
}
process.exit(ok ? 0 : 1);
